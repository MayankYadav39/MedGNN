from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from utils.XAIUtilities import explain_prediction_and_uncertainty, get_top_features, get_clinical_mapping, decompose_evidence
from utils.VisualXAI import generate_clinical_report

warnings.filterwarnings("ignore")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _build_model(self):
        # model input depends on data
        # train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = test_data.max_seq_len  # redefine seq_len
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.enc_in = test_data.X.shape[2]  # redefine enc_in
        self.args.num_class = len(np.unique(test_data.y))
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # Use mc_samples=1 to force single forward pass (no uncertainty during validation)
                if self.swa:
                    outputs = self.swa_model(batch_x, padding_mask, None, None, mc_samples=1)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None, mc_samples=1)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        # print(trues_onehot.shape)
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        print(train_data.X.shape)
        print(train_data.y.shape)
        print(vali_data.X.shape)
        print(vali_data.y.shape)
        print(test_data.X.shape)
        print(test_data.y.shape)

        path = (
            "./checkpoints/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param
        print(f"Total Trainable Params: {total_params}")


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                
                if isinstance(outputs, tuple):
                    # Heteroscedastic Loss for Aleatoric Uncertainty
                    logits, log_var = outputs
                    # Sample noise: z = mu + eps * exp(0.5 * log_var)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn_like(std)
                    z = logits + eps * std
                    loss = criterion(z, label.long())
                    
                    # Add KL divergence loss for Bayesian weights
                    kl_loss = self.model.get_kl_loss()
                    kl_weight = self.args.batch_size / len(train_loader.dataset)
                    loss = loss + kl_weight * kl_loss
                else:
                    loss = criterion(outputs, label.long())
                
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            self.swa_model.update_parameters(self.model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {test_metrics_dict['Precision']:.5f}, "
                f"Recall: {test_metrics_dict['Recall']:.5f} "
                f"F1: {test_metrics_dict['F1']:.5f}, "
                f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
            )
            early_stopping(
                -val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            """if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)"""

        best_model_path = path + "checkpoint.pth"
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path, map_location='cuda'))
        else:
            self.model.load_state_dict(torch.load(best_model_path, map_location='cuda'))

        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            path = (
                "./checkpoints/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path, map_location='cuda'))
            else:
                self.model.load_state_dict(torch.load(model_path, map_location='cuda'))

        criterion = self._select_criterion()
        
        # Check if MC Dropout is enabled
        use_mc = getattr(self.args, 'enable_mc_dropout', False)
        mc_samples = getattr(self.args, 'mc_samples', 10) if use_mc else 1
        
        if use_mc:
            print(f"\nUsing MC Dropout with {mc_samples} samples for uncertainty estimation")
            # Run test with MC Dropout
            test_loss, test_metrics_dict, uncertainty_metrics = self._test_with_uncertainty(
                test_data, test_loader, criterion, mc_samples
            )
            # For validation, use single pass (faster)
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        else:
            # Standard testing without uncertainty
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)
            uncertainty_metrics = None

        # result save
        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # # save the adjacency matrix
        # adj_path = (
        #         "./adj_weight/"
        #         + self.args.task_name
        #         + "/"
        #         + self.args.model_id
        #         + "/"
        #         + self.args.model
        #         + "/"
        # )
        # if not os.path.exists(adj_path):
        #     os.makedirs(adj_path)

        # for l in range(len(adjacency_matrix_list)):
        #     matrix = adjacency_matrix_list[l].detach().cpu().numpy()
        #     np.save(adj_path + 'matrix_{}.npy'.format(l), matrix)


        print(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        
        # Add uncertainty metrics if available
        if uncertainty_metrics is not None:
            uncertainty_str = (
                f"Uncertainty Metrics:\n"
                f"  Mean Variance: {uncertainty_metrics['mean_variance']:.5f}\n"
                f"  Mean Predictive Entropy: {uncertainty_metrics['mean_entropy']:.5f}\n"
            )
            
            if 'mean_epistemic' in uncertainty_metrics:
                uncertainty_str += (
                    f"  Mean Epistemic (Model): {uncertainty_metrics['mean_epistemic']:.5f}\n"
                    f"  Mean Aleatoric (Data): {uncertainty_metrics['mean_aleatoric']:.5f}\n"
                )
            
            uncertainty_str += (
                f"  High Uncertainty Samples (>90th percentile): {uncertainty_metrics['high_uncertainty_count']} "
                f"({uncertainty_metrics['high_uncertainty_pct']:.2f}%)\n"
                f"  Uncertainty-Error Correlation: {uncertainty_metrics['uncertainty_error_corr']:.4f}\n"
            )
            print(uncertainty_str)
            f.write(uncertainty_str)
            
            # Save detailed uncertainty data
            uncertainty_path = folder_path + 'uncertainty_data.npz'
            np.savez(
                uncertainty_path,
                predictions=uncertainty_metrics['predictions'],
                true_labels=uncertainty_metrics['true_labels'],
                variance=uncertainty_metrics['variance'],
                entropy=uncertainty_metrics['entropy'],
                epistemic=uncertainty_metrics['epistemic'],
                aleatoric=uncertainty_metrics['aleatoric'],
            )
            print(f"Saved uncertainty data to {uncertainty_path}")
        
        f.write("\n")
        f.write("\n")
        f.close()
        return
    
    def _test_with_uncertainty(self, test_data, test_loader, criterion, mc_samples):
        """Test with MC Dropout uncertainty estimation"""
        total_loss = []
        all_preds = []
        all_trues = []
        all_variances = []
        all_entropies = []
        all_epistemic = []
        all_aleatoric = []
        
        model_to_use = self.swa_model if self.swa else self.model
        model_to_use.eval()
        
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                # Get predictions with uncertainty
                # Return can be (mean, total_unc, epistemic, aleatoric) or (mean, variance)
                out = model_to_use(batch_x, padding_mask, None, None, mc_samples=mc_samples)
                
                if len(out) == 4:
                    output, variance, epistemic, aleatoric = out
                else:
                    output, variance = out
                    epistemic = variance
                    aleatoric = torch.zeros_like(variance)
                
                # Compute loss on mean prediction
                loss = criterion(output.cpu(), label.long().cpu())
                total_loss.append(loss.item())
                
                # Compute predictive entropy
                probs = torch.nn.functional.softmax(output, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                
                all_preds.append(output.detach())
                all_trues.append(label)
                all_variances.append(variance.detach())
                all_entropies.append(entropy.detach())
                
                if len(out) == 4:
                    all_epistemic.append(epistemic.detach())
                    all_aleatoric.append(aleatoric.detach())
        
        total_loss = np.mean(total_loss)
        
        # Concatenate all batches
        preds = torch.cat(all_preds, 0)
        trues = torch.cat(all_trues, 0)
        variances = torch.cat(all_variances, 0).cpu().numpy()
        entropies = torch.cat(all_entropies, 0).cpu().numpy()
        
        if all_epistemic:
            epistemic_np = torch.cat(all_epistemic, 0).cpu().numpy()
            aleatoric_np = torch.cat(all_aleatoric, 0).cpu().numpy()
        else:
            epistemic_np = variances
            aleatoric_np = np.zeros_like(variances)
        
        # Compute classification metrics
        probs = torch.nn.functional.softmax(preds, dim=-1)
        trues_onehot = torch.nn.functional.one_hot(
            trues.reshape(-1,).to(torch.long),
            num_classes=self.args.num_class,
        ).float().cpu().numpy()
        
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        probs_np = probs.cpu().numpy()
        trues_np = trues.flatten().cpu().numpy()
        
        metrics_dict = {
            "Accuracy": accuracy_score(trues_np, predictions),
            "Precision": precision_score(trues_np, predictions, average="macro"),
            "Recall": recall_score(trues_np, predictions, average="macro"),
            "F1": f1_score(trues_np, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs_np, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs_np, average="macro"),
        }
        
        # Compute uncertainty metrics
        errors = (predictions != trues_np).astype(float)
        threshold_90 = np.percentile(variances, 90)
        high_uncertainty_mask = variances > threshold_90
        
        uncertainty_metrics = {
            'mean_variance': np.mean(variances),
            'mean_entropy': np.mean(entropies),
            'mean_epistemic': np.mean(epistemic_np),
            'mean_aleatoric': np.mean(aleatoric_np),
            'high_uncertainty_count': int(high_uncertainty_mask.sum()),
            'high_uncertainty_pct': 100.0 * high_uncertainty_mask.sum() / len(variances),
            'uncertainty_error_corr': np.corrcoef(variances, errors)[0, 1],
            'predictions': predictions,
            'true_labels': trues_np,
            'variance': variances,
            'entropy': entropies,
            'epistemic': epistemic_np,
            'aleatoric': aleatoric_np,
        }
        
        model_to_use.train()
        return total_loss, metrics_dict, uncertainty_metrics

    def explain(self, setting, sample_indices=None):
        """
        Generate explanations for specific samples (or top-uncertainty samples).
        """
        _, test_loader = self._get_data(flag="TEST")
        model = self.swa_model if self.swa else self.model
        model.eval()

        print(f"\nGenerating explanations for uncertainty and predictions...")
        
        # We need uncertainty_data.npz to find high-uncertainty samples if indices not provided
        folder_path = f"./results/{self.args.task_name}/{self.args.model_id}/{self.args.model}/"
        data_path = os.path.join(folder_path, 'uncertainty_data.npz')
        
        if sample_indices is None and os.path.exists(data_path):
            data = np.load(data_path)
            variance = data['variance']
            # Get top 5 most uncertain samples
            sample_indices = np.argsort(variance)[-5:].tolist()
            print(f"No indices provided. Explaining top 5 most uncertain samples: {sample_indices}")
        elif sample_indices is None:
            sample_indices = [0] # Default to first sample
            print(f"No uncertainty data found. Explaining sample 0.")

        all_explanations = []

        # Collect samples from loader
        collected_samples = []
        collected_labels = []
        collected_masks = []
        
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                collected_samples.append(batch_x)
                collected_labels.append(label)
                collected_masks.append(padding_mask)
                if sum([s.size(0) for s in collected_samples]) > max(sample_indices):
                    break
        
        X = torch.cat(collected_samples, 0)
        Y = torch.cat(collected_labels, 0)
        M = torch.cat(collected_masks, 0)

        for idx in sample_indices:
            x = X[idx:idx+1].to(self.device).requires_grad_(True)
            y = Y[idx].item()
            
            # 1. Generate Attribution
            pred_attr, unc_attr = explain_prediction_and_uncertainty(
                model, x, mc_samples=getattr(self.args, 'mc_samples', 10)
            )
            
            # 2. Extract top features
            # Get feature names if available (defaulting to indices)
            # 2. Extract top features
            # Get clinical feature names
            feat_dim = x.shape[2]
            feature_names = get_clinical_mapping(self.args.data, feat_dim)
            
            top_pred = get_top_features(pred_attr, feature_names=feature_names, top_k=5)
            top_unc = get_top_features(unc_attr, feature_names=feature_names, top_k=5)
            
            # Reasoning (+1 and -1)
            pos_evidence, neg_evidence = decompose_evidence(pred_attr)
            
            # 3. Generate Visual Report
            report_dir = os.path.join(folder_path, f"explanations/sample_{idx}")
            # Get prediction confidence from uncertainty metadata if available
            confidence = 100.0 # Placeholder if not found
            
            pred_img, unc_img = generate_clinical_report(
                idx, int(y), confidence, 
                pred_attr[0], unc_attr[0], x[0].detach().cpu().numpy(), 
                report_dir
            )
            
            explanation = {
                'sample_index': idx,
                'true_label': int(y),
                'top_prediction_features': top_pred,
                'top_uncertainty_features': top_unc,
                'reasoning': {
                    'supporting_evidence': get_top_features(pos_evidence.reshape(1, 1, -1), feature_names=feature_names),
                    'contradicting_evidence': get_top_features(neg_evidence.reshape(1, 1, -1), feature_names=feature_names)
                },
                'visual_reports': {
                    'prediction_reason': pred_img,
                    'uncertainty_reason': unc_img
                }
            }
            all_explanations.append(explanation)
            
            print(f"\nSample {idx} (Label {y}):")
            print("  Top features for Prediction (Cumulative):")
            for f in top_pred:
                print(f"    - {f['name']}: {f['importance']:.4f}")
            
            print("  Reasoning (+ vs -):")
            print("    Positive (Supporting):")
            for f in explanation['reasoning']['supporting_evidence']:
                print(f"      - {f['name']}: +{f['importance']:.4f}")
            print("    Negative (Contradicting):")
            for f in explanation['reasoning']['contradicting_evidence']:
                print(f"      - {f['name']}: -{f['importance']:.4f}")
                
            print("  Top features for Uncertainty:")
            for f in top_unc:
                print(f"    - {f['name']}: {f['importance']:.4f}")

        # Save explanations
        save_path = os.path.join(folder_path, 'explanations.json')
        import json
        with open(save_path, 'w') as f:
            json.dump(all_explanations, f, indent=4)
        print(f"\nExplanations saved to {save_path}")
        
        return all_explanations
