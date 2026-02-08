"""Test adaptive likelihood scaling at high SNR"""
import torch
import numpy as np
from inference_methods import (
    least_squares_baseline,
    regular_inference,
    split_gibbs_inference,
)

# Load model
print("Loading model...")
checkpoint = torch.load("mimo_sgdd_model.pth", map_location="cuda")
model_state = checkpoint["model_state"]

from MIMO_score_based_diffusion import TransformerScoreModel

model = TransformerScoreModel(
    embedding_dim=256, 
    num_layers=6, 
    num_heads=8,
    seq_len=1560
)
model.load_state_dict(model_state)
model = model.cuda()
model.eval()

torch.manual_seed(42)
np.random.seed(42)

# Test high SNR scenarios
snr_levels = [5, 10, 15, 20]
num_trials = 3
num_bits = 1560

print(f"\n{'='*90}")
print(f"HIGH SNR COMPARISON: Fixed vs Adaptive Likelihood Scaling")
print(f"{'='*90}")

for snr_db in snr_levels:
    snr_linear = 10 ** (snr_db / 10.0)
    sigma_noise = np.sqrt(1.0 / snr_linear)

    results_fixed = {'ls': [], 'regular': [], 'split_gibbs': []}
    results_adaptive = {'ls': [], 'regular': [], 'split_gibbs': []}

    print(f"\n--- SNR = {snr_db:+.0f} dB ---")

    for trial in range(num_trials):
        # Generate test data
        x_true = torch.randint(0, 2, (1, num_bits), dtype=torch.float32)
        num_frames = num_bits // 2
        H = (torch.randn(1, 2, 2, num_frames) + 1j * torch.randn(1, 2, 2, num_frames)) / np.sqrt(2)
        x_real = (2 * x_true - 1).view(1, 2, num_frames).to(torch.complex64)
        y = torch.einsum('bijk, bjk -> bik', H, x_real)
        y = y + sigma_noise * (torch.randn_like(y) + 1j * torch.randn_like(y)) / np.sqrt(2)
        
        x_true_cuda = x_true.cuda()
        y_cuda = y.cuda()
        H_cuda = H.cuda()
        
        # Method 1: Least Squares
        x_pred_ls = least_squares_baseline(y_cuda, H_cuda)
        ber_ls = torch.mean(((x_pred_ls > 0.5) != x_true_cuda).float()).item()
        results_fixed['ls'].append(ber_ls)
        results_adaptive['ls'].append(ber_ls)
        
        # Method 2: Regular Inference
        x_pred_reg = regular_inference(
            model=model,
            y_rx=y_cuda,
            h_channel=H_cuda,
            device='cuda',
            num_steps=15
        )
        ber_reg = torch.mean(((x_pred_reg > 0.5) != x_true_cuda).float()).item()
        results_fixed['regular'].append(ber_reg)
        results_adaptive['regular'].append(ber_reg)
        
        # Method 3a: Split Gibbs with FIXED likelihood scale
        likelihood_scale_fixed = 0.01
        x_pred_sg_fixed = split_gibbs_inference(
            model=model,
            y_rx=y_cuda,
            h_channel=H_cuda,
            device='cuda',
            num_diffusion_steps=15,
            num_mh_steps=100,
            snr_db=snr_db,
            likelihood_scale=likelihood_scale_fixed
        )
        ber_sg_fixed = torch.mean(((x_pred_sg_fixed > 0.5) != x_true_cuda).float()).item()
        results_fixed['split_gibbs'].append(ber_sg_fixed)
        
        # Method 3b: Split Gibbs with ADAPTIVE likelihood scale
        # Scale grows with SNR: at -10dB ~0.01, at +20dB ~1.0
        likelihood_scale_adaptive = max(0.01, 10.0 ** (snr_db / 10.0) * 0.01)
        x_pred_sg_adaptive = split_gibbs_inference(
            model=model,
            y_rx=y_cuda,
            h_channel=H_cuda,
            device='cuda',
            num_diffusion_steps=15,
            num_mh_steps=100,
            snr_db=snr_db,
            likelihood_scale=likelihood_scale_adaptive
        )
        ber_sg_adaptive = torch.mean(((x_pred_sg_adaptive > 0.5) != x_true_cuda).float()).item()
        results_adaptive['split_gibbs'].append(ber_sg_adaptive)
        
        print(f"  Trial {trial + 1}: LS={ber_ls:.6f}, Reg={ber_reg:.6f}, SG_Fixed={ber_sg_fixed:.6f}, SG_Adaptive={ber_sg_adaptive:.6f} (scale={likelihood_scale_adaptive:.3f})")

    # Summary
    print(f"  FIXED scaling (0.01):")
    print(f"    LS:        {np.mean(results_fixed['ls']):.6f}")
    print(f"    Regular:   {np.mean(results_fixed['regular']):.6f}")
    print(f"    SG Fixed:  {np.mean(results_fixed['split_gibbs']):.6f}")
    
    print(f"  ADAPTIVE scaling:")
    print(f"    LS:        {np.mean(results_adaptive['ls']):.6f}")
    print(f"    Regular:   {np.mean(results_adaptive['regular']):.6f}")
    print(f"    SG Adapt:  {np.mean(results_adaptive['split_gibbs']):.6f}  ← Improvement: {100*(np.mean(results_fixed['split_gibbs'])-np.mean(results_adaptive['split_gibbs']))/np.mean(results_fixed['split_gibbs']):+.1f}%")

print(f"\n{'='*90}\n")
