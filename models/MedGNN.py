import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import Multi_Resolution_Data, Frequency_Embedding
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FormerLayer, DifferenceFormerlayer
from layers.Multi_Resolution_GNN import MRGNN
from layers.Difference_Pre import DifferenceDataEmb, DataRestoration
from layers.BayesianLayers import BayesianLinear, AleatoricLinear


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout
        self.output_attention = configs.output_attention
        self.activation = configs.activation
        self.resolution_list = list(map(int, configs.resolution_list.split(",")))
        self.enable_bayesian = getattr(configs, 'enable_bayesian', False)


        self.res_num = len(self.resolution_list)
        self.stride_list = self.resolution_list
        self.res_len = [int(self.seq_len//res)+1 for res in self.resolution_list]
        self.augmentations = configs.augmentations.split(",")
        
        # Monte Carlo Dropout configuration
        self.enable_mc_dropout = getattr(configs, 'enable_mc_dropout', False)
        self.mc_samples = getattr(configs, 'mc_samples', 10)

        # step1: multi_resolution_data
        self.multi_res_data = Multi_Resolution_Data(self.enc_in, self.resolution_list, self.stride_list)

        # step2.1: frequency convolution network
        self.freq_embedding = Frequency_Embedding(self.d_model, self.res_len, self.augmentations)

        # step2.2: difference attention network
        self.diff_data_emb = DifferenceDataEmb(self.res_num, self.enc_in, self.d_model)
        self.difference_attention = Encoder(
            [
                EncoderLayer(
                    DifferenceFormerlayer(
                        self.enc_in,
                        self.res_num,
                        self.d_model,
                        self.n_heads,
                        self.dropout,
                        self.output_attention,
                        use_mc_dropout=self.enable_mc_dropout,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    use_mc_dropout=self.enable_mc_dropout,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.data_restoration = DataRestoration(self.res_num, self.enc_in, self.d_model)
        self.embeddings = nn.ModuleList([nn.Linear(res_len, self.d_model) for res_len in self.res_len])

        # step 3: transformer
        self.encoder = Encoder(
            [
                EncoderLayer(
                    FormerLayer(
                        len(self.resolution_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                        use_mc_dropout=self.enable_mc_dropout,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    use_mc_dropout=self.enable_mc_dropout,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # step 4: multi-resolution GNN
        self.mrgnn = MRGNN(configs, self.res_len)

        # step 5: projection
        if self.enable_bayesian:
            self.projection = AleatoricLinear(self.d_model * self.enc_in, configs.num_class)
        else:
            self.projection = nn.Linear(self.d_model * self.enc_in, configs.num_class)


    def get_kl_loss(self):
        """Compute KL divergence for all Bayesian layers"""
        kl_loss = 0
        for m in self.modules():
            if isinstance(m, BayesianLinear):
                kl_loss += m.kl_divergence()
        return kl_loss


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, mc_samples=None):
        """
        Forward pass with optional MC Dropout for uncertainty estimation.
        
        Args:
            mc_samples: Number of MC samples. If None, uses 1 during training, config value during eval.
                       If 1, single forward pass. If > 1, MC dropout with multiple passes.
        
        Returns:
            If mc_samples == 1 or training: output logits (B, num_class)
            If mc_samples > 1 and eval: tuple of (mean_output, uncertainty_variance)
        """
        # ALWAYS use single forward pass during training
        if self.training:
            return self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        
        # Determine number of MC samples for inference
        if mc_samples is None:
            mc_samples = self.mc_samples if self.enable_mc_dropout else 1
        
        # Single forward pass (non-MC inference)
        if mc_samples == 1 or not self.enable_mc_dropout:
            return self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        
        # Multiple forward passes for uncertainty estimation
        return self._mc_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, mc_samples)
    
    def _single_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """Single forward pass"""
        B, T, C = x_enc.shape

        # step1: multi_resolution_data
        multi_res_data = self.multi_res_data(x_enc)

        # step2.1: frequency convolution network
        enc_out_1 = self.freq_embedding(multi_res_data)

        # step2.2: difference attention network
        x_diff_emb, x_padding = self.diff_data_emb(multi_res_data)
        x_diff_enc, attns = self.difference_attention(x_diff_emb, attn_mask=None)
        enc_out_2 = self.data_restoration(x_diff_enc, x_padding)
        enc_out_2 = [self.embeddings[l](enc_out_2[l]) for l in range(self.res_num)]

        # step 3: transformer
        data_enc = [enc_out_1[l] + enc_out_2[l] for l in range(self.res_num)]
        enc_out, attns = self.encoder(data_enc, attn_mask=None)

        # step 4: multi-resolution GNN
        output, adjacency_matrix_list = self.mrgnn(enc_out)

        # step 5: projection
        output = output.reshape(B, -1)
        output = self.projection(output)

        return output
    
    def _mc_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask, mc_samples):
        """Multiple forward passes for MC Dropout and Bayesian uncertainty estimation"""
        means = []
        log_vars = []
        
        # Set to eval mode (MC Dropout will still be active)
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            for _ in range(mc_samples):
                out = self._single_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
                if self.enable_bayesian:
                    logits, log_var = out
                    means.append(logits)
                    log_vars.append(log_var)
                else:
                    means.append(out)
        
        # Restore training mode if needed
        if was_training:
            self.train()
        
        # Stack predictions: (mc_samples, B, num_class)
        all_means = torch.stack(means, dim=0)
        mean_pred = all_means.mean(dim=0)  # (B, num_class)
        
        # Epistemic Uncertainty: Variance of the mean predictions
        epistemic_unc = all_means.var(dim=0).mean(dim=1)  # (B,)
        
        if self.enable_bayesian:
            # Aleatoric Uncertainty: Mean of the predicted variances
            all_log_vars = torch.stack(log_vars, dim=0)
            all_vars = torch.exp(all_log_vars)
            aleatoric_unc = all_vars.mean(dim=0).mean(dim=1)  # (B,)
            
            # Total Uncertainty
            total_unc = epistemic_unc + aleatoric_unc
            return mean_pred, total_unc, epistemic_unc, aleatoric_unc
        
        return mean_pred, epistemic_unc