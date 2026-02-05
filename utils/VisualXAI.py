import matplotlib.pyplot as plt
import numpy as np
import os

def plot_uncertainty_attribution(time_series, attribution, title="Uncertainty Attribution", save_path=None):
    """
    Plots a time-series with an attribution heatmap overlay.
    
    Args:
        time_series: (T, C) array of signal values
        attribution: (T, C) array of importance scores
        title: plot title
        save_path: path to save the image
    """
    T, C = time_series.shape
    fig, axes = plt.subplots(C, 1, figsize=(12, 2 * C), sharex=True)
    
    if C == 1:
        axes = [axes]
        
    for i in range(C):
        ax = axes[i]
        # Plot raw signal
        ax.plot(time_series[:, i], color='black', alpha=0.3, label='Signal')
        
        # Plot heatmap using attribution
        # Normalize attribution for plotting
        attr = attribution[:, i]
        attr_norm = (attr - np.min(attr)) / (np.max(attr) - np.min(attr) + 1e-8)
        
        # Highlight points with high attribution
        points = np.arange(T)
        ax.scatter(points, time_series[:, i], c=attr_norm, cmap='hot', s=10, label='Importance')
        
        ax.set_ylabel(f'Ch {i}')
        if i == 0:
            ax.set_title(title)
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close()

def generate_clinical_report(sample_idx, prediction, confidence, pred_attr, unc_attr, signal, save_dir):
    """
    Generates a visual summary report for a specific sample.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 1. Plot Prediction Attribution (Why this result?)
    pred_path = os.path.join(save_dir, f"sample_{sample_idx}_prediction_reason.png")
    plot_uncertainty_attribution(signal, pred_attr, title=f"Sample {sample_idx}: Prediction Reason (Class {prediction})", save_path=pred_path)
    
    # 2. Plot Uncertainty Attribution (Why uncertain?)
    unc_path = os.path.join(save_dir, f"sample_{sample_idx}_uncertainty_reason.png")
    plot_uncertainty_attribution(signal, unc_attr, title=f"Sample {sample_idx}: Uncertainty Reason (Conf: {confidence:.2f}%)", save_path=unc_path)
    
    return pred_path, unc_path
