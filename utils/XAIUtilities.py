import torch
import torch.nn.functional as F
import numpy as np

def integrated_gradients(model, input_tensor, target_class=None, baseline=None, steps=50, mc_samples=1, attribute_to_variance=False):
    """
    Implements Integrated Gradients for feature attribution.
    Can attribute either to a specific class probability or to the prediction variance (uncertainty).
    
    Args:
        model: The torch model
        input_tensor: (B, T, C) input
        target_class: index of the class to explain (if None, use predicted class)
        baseline: baseline input (usually zeros)
        steps: number of interpolation steps
        mc_samples: number of MC samples if uncertainty attribution is requested
        attribute_to_variance: If True, attribute to the variance (uncertainty) instead of prediction
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
        
    input_tensor.requires_grad = True
    
    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, steps).to(input_tensor.device)
    interpolated_inputs = [baseline + alpha * (input_tensor - baseline) for alpha in alphas]
    
    grads = []
    
    for x in interpolated_inputs:
        x = x.clone().detach().requires_grad_(True)
        
        # Determine what to compute gradients for
        if attribute_to_variance:
            # For uncertainty attribution, we need multiple samples and compute grad of variance
            model.eval()
            preds = []
            for _ in range(mc_samples):
                # Standard MedGNN forward requires padding_mask, x_dec, x_mark_dec
                # Assuming empty/none for these during attribution
                out = model(x, None, None, None, mc_samples=1)
                if isinstance(out, tuple):
                    out = out[0]
                preds.append(out)
            
            all_preds = torch.stack(preds)
            # Variance across samples
            target_val = all_preds.var(dim=0).mean() # Scalar summary of uncertainty
        else:
            # Standard prediction attribution
            out = model(x, None, None, None, mc_samples=1)
            if isinstance(out, tuple):
                out = out[0]
            
            if target_class is None:
                target_class = out.argmax(dim=-1)
                
            probs = F.softmax(out, dim=-1)
            target_val = probs[0, target_class] # Explaining the target class probability
            
        model.zero_grad()
        target_val.backward()
        grads.append(x.grad.detach())
        
    avg_grads = torch.stack(grads).mean(dim=0)
    integrated_grad = (input_tensor - baseline) * avg_grads
    
    return integrated_grad.detach().cpu().numpy()

def decompose_evidence(attribution, top_k=5):
    """
    Splits attribution into positive (supporting) and negative (contradicting) evidence.
    """
    if len(attribution.shape) == 3:
        attribution = attribution[0]
        
    pos_evidence = np.maximum(attribution, 0)
    neg_evidence = np.minimum(attribution, 0)
    
    # Aggregate over time
    pos_sum = np.sum(pos_evidence, axis=0)
    neg_sum = np.abs(np.sum(neg_evidence, axis=0))
    
    return pos_sum, neg_sum

def explain_prediction_and_uncertainty(model, input_tensor, target_class=None, mc_samples=10, steps=50):
    """
    Computes attributions for both the prediction and the uncertainty.
    """
    # 1. Explain Prediction
    prediction_attribution = integrated_gradients(
        model, input_tensor, target_class=target_class, 
        steps=steps, mc_samples=1, attribute_to_variance=False
    )
    
    # 2. Explain Uncertainty
    uncertainty_attribution = integrated_gradients(
        model, input_tensor, steps=steps, 
        mc_samples=mc_samples, attribute_to_variance=True
    )
    
    return prediction_attribution, uncertainty_attribution

def get_top_features(attribution, feature_names=None, top_k=5):
    """
    Aggregates attribution scores across time and returns top features.
    Attribution shape: (1, T, C) or (T, C)
    """
    if len(attribution.shape) == 3:
        attribution = attribution[0]
        
    # Aggregate importance over time (sum of absolute values)
    feature_importance = np.sum(np.abs(attribution), axis=0)
    
    top_indices = np.argsort(feature_importance)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        name = feature_names[idx] if feature_names else f"Feature {idx}"
        results.append({
            'index': int(idx),
            'name': name,
            'importance': float(feature_importance[idx])
        })
        
    return results

def get_clinical_mapping(dataset_name, num_channels):
    """
    Provides human-readable names for features based on the dataset modality.
    """
    # ECG/Vitals Mapping (Example for APAVA)
    apava_mapping = {
        0: "ECG Lead I", 1: "ECG Lead II", 2: "Respiration", 3: "SPO2",
        4: "Heart Rate", 5: "Systolic BP", 6: "Diastolic BP", 7: "MAP",
        8: "Temperature", 9: "Pulse", 10: "CO2", 11: "Glucose",
        12: "O2 Flow", 13: "Respiratory Rate", 14: "ST Segment", 15: "QT Interval"
    }
    
    # EEG 10-20 System Mapping
    eeg_mapping = {
        0: "Fp1", 1: "Fp2", 2: "F7", 3: "F3", 4: "Fz", 5: "F4", 6: "F8",
        7: "T7", 8: "C3", 9: "Cz", 10: "C4", 11: "T8", 12: "P7", 13: "P3",
        14: "Pz", 15: "P4", 16: "P8", 17: "O1", 18: "O2", 19: "T5", 20: "T6", 21: "A1"
    }
    
    if "APAVA" in dataset_name.upper():
        mapping = apava_mapping
    elif "EEG" in dataset_name.upper() or "BRAIN" in dataset_name.upper():
        mapping = eeg_mapping
    else:
        mapping = {}
        
    return [mapping.get(i, f"Channel {i}") for i in range(num_channels)]
