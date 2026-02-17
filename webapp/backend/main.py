from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import os
import sys
from pydantic import BaseModel
from typing import List, Optional

# Add the parent directory to sys.path to import MedGNN modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import MedGNN
from utils.XAIUtilities import (
    explain_prediction_and_uncertainty, 
    get_top_features, 
    get_clinical_mapping, 
    decompose_evidence
)

app = FastAPI(title="MedGNN Clinical Dashboard API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration class to match the model expectations
class Configs:
    def __init__(self, enc_in=16, seq_len=256, num_class=2):
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.d_model = 256
        self.d_ff = 512
        self.n_heads = 8
        self.e_layers = 4
        self.dropout = 0.1
        self.output_attention = False
        self.activation = 'gelu'
        self.resolution_list = "2,4,6,8"
        self.nodedim = 10
        self.num_class = num_class
        self.augmentations = "none"
        self.enable_mc_dropout = True
        self.mc_samples = 10
        self.enable_bayesian = True
        self.task_name = "classification"
        self.model = "MedGNN"
        self.model_id = "APAVA-Subject"

# Global model variable
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def load_model():
    global model
    try:
        configs = Configs()
        configs.enable_bayesian = True
        print(f"\n[BACKEND] Initializing MedGNN Model...")
        print(f"[BACKEND] Configs: num_class={configs.num_class}, bayesian={configs.enable_bayesian}")
        
        model = MedGNN.Model(configs).to(device)
        
        # Load the trained weights
        checkpoint_dir = "../../checkpoints/classification/APAVA-Subject/MedGNN"
        if os.path.exists(checkpoint_dir):
            subfolders = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
            if subfolders:
                latest_folder = max(subfolders, key=os.path.getmtime)
                checkpoint_path = os.path.join(latest_folder, "checkpoint.pth")
                if os.path.exists(checkpoint_path):
                    # Robust loading: catch mismatch
                    try:
                        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                        print(f"[BACKEND] Success: Loaded weights from {checkpoint_path}")
                    except Exception as load_err:
                        print(f"[BACKEND] WARNING: Weight mismatch! {str(load_err)}")
                        print(f"[BACKEND] Falling back to initialized Bayesian weights.")
        
        model.eval()
        print("[BACKEND] Dashboard API is Ready on port 8000")
        
    except Exception as e:
        print(f"[BACKEND] CRITICAL ERROR during startup: {str(e)}")

# Clinical diagnosis mapper (APAVA: 0=AD, 1=HC)
DIAGNOSIS_MAP = {
    0: {"name": "Alzheimer's (AD)", "status": "Positive", "color": "#f43f5e"},
    1: {"name": "Healthy Control (HC)", "status": "Negative", "color": "#10b981"}
}

class InferenceRequest(BaseModel):
    data: List[List[List[float]]] # (B, T, C)
    data_type: str = "APAVA"

@app.get("/sample/{index}")
async def get_sample(index: int):
    """Fetch a specific sample from the TEST dataset"""
    print(f"[BACKEND] Fetching sample {index}...")
    # Create data provider for TEST flag
    from data_provider.data_factory import data_provider
    configs = Configs()
    # Mocking args for data_provider
    class Args:
        def __init__(self, c):
            self.data = "APAVA"
            self.root_path = "../../../captum_env/dataset/APAVA/"
            self.batch_size = 1
            self.num_workers = 0
            self.seq_len = c.seq_len
            self.enc_in = c.enc_in
            self.num_class = c.num_class
            self.resolution_list = c.resolution_list
            self.augmentations = "none"
            self.seed = 41
            self.task_name = "classification"
            self.single_channel = False
            self.embed = "none"
            self.freq = "h"
    
    args = Args(configs)
    try:
        test_data, test_loader = data_provider(args, flag='TEST')
        if index < 0 or index >= len(test_data):
            raise HTTPException(status_code=404, detail=f"Sample index {index} out of range (max {len(test_data)-1})")
        
        # Get specific sample (Unpack only 2 values as per APAVALoader)
        x, y = test_data[index]
        return {
            "index": index,
            "data": x.tolist(),
            "label": int(y)
        }
    except Exception as e:
        print(f"[BACKEND] Error in get_sample: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: InferenceRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        x = torch.tensor(request.data).float().to(device)
        B, T, C = x.shape
        
        # Run MC Inference
        with torch.no_grad():
            # (mean_pred, total_unc, epistemic, aleatoric)
            out = model(x, None, None, None, mc_samples=10)
            
            if isinstance(out, tuple) and len(out) == 4:
                logits, total_unc, epistemic, aleatoric = out
            elif isinstance(out, tuple) and len(out) == 2:
                logits, total_unc = out
                epistemic = total_unc
                aleatoric = torch.zeros_like(total_unc)
            else:
                logits = out
                total_unc = torch.zeros(B)
                epistemic = torch.zeros(B)
                aleatoric = torch.zeros(B)
                
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1)
            
            # Debug: Print actual uncertainty values
            print(f"[BACKEND DEBUG] Total uncertainty: {total_unc.tolist()}")
            print(f"[BACKEND DEBUG] Epistemic: {epistemic.tolist()}")
            print(f"[BACKEND DEBUG] Aleatoric: {aleatoric.tolist()}")
            
        return {
            "prediction": prediction.tolist(),
            "probabilities": probs.tolist(),
            "uncertainty": {
                "total": total_unc.tolist(),
                "epistemic": epistemic.tolist(),
                "aleatoric": aleatoric.tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain(request: InferenceRequest, target_class: Optional[int] = None):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        x = torch.tensor(request.data).float().to(device)
        
        # Generate attributions
        pred_attr, unc_attr = explain_prediction_and_uncertainty(
            model, x, mc_samples=10, steps=20
        )
        
        # Determine feature names
        feat_dim = x.shape[2]
        feature_names = get_clinical_mapping(request.data_type if hasattr(request, 'data_type') else "APAVA", feat_dim)
        
        top_pred = get_top_features(pred_attr, feature_names=feature_names, top_k=5)
        top_unc = get_top_features(unc_attr, feature_names=feature_names, top_k=5)
        
        # Decompose evidence
        pos_ev, neg_ev = decompose_evidence(pred_attr)
        
        # Debug: Print attribution shapes
        print(f"[BACKEND DEBUG] pred_attr shape: {pred_attr.shape}")
        print(f"[BACKEND DEBUG] unc_attr shape: {unc_attr.shape}")
        print(f"[BACKEND DEBUG] unc_attr sample values: {unc_attr[0, :5, :].tolist()}")  # First 5 timesteps
        
        return {
            "prediction_attribution": pred_attr.tolist(),
            "uncertainty_attribution": unc_attr.tolist(),
            "top_features_prediction": top_pred,
            "top_features_uncertainty": top_unc,
            "reasoning": {
                "supporting_evidence": get_top_features(pos_ev.reshape(1, 1, -1), feature_names=feature_names),
                "contradicting_evidence": get_top_features(neg_ev.reshape(1, 1, -1), feature_names=feature_names)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
