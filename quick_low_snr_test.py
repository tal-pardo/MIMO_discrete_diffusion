"""Quick single-scenario test at low SNR for all three methods"""
import torch
import numpy as np
from inference_methods import (
    least_squares_baseline,
    regular_inference,
    split_gibbs_inference,
    MIMOLikelihood,
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

# Single low SNR scenario
snr_db = -10.0
snr_linear = 10 ** (snr_db / 10.0)
sigma_noise = np.sqrt(1.0 / snr_linear)

print(f"\n{'='*60}")
print(f"QUICK LOW-SNR TEST AT SNR = {snr_db:.1f} dB")
print(f"{'='*60}")

num_trials = 3
num_bits = 1560

results = {
    'ls': [],
    'regular': [],
    'split_gibbs': []
}

for trial in range(num_trials):
    print(f"\nTrial {trial + 1}/{num_trials}:")
    
    # Generate test data using the proper MIMO simulation
    x_true = torch.randint(0, 2, (1, num_bits), dtype=torch.float32)
    
    # Create proper MIMO channel: (batch, 2, 2, num_frames)
    num_frames = num_bits // 2
    H = (torch.randn(1, 2, 2, num_frames) + 1j * torch.randn(1, 2, 2, num_frames)) / np.sqrt(2)
    
    # Convert bits to BPSK and reshape to (batch, 2, num_frames)
    x_real = (2 * x_true - 1).view(1, 2, num_frames).to(torch.complex64)
    
    # Compute y = H @ x + noise
    y = torch.einsum('bijk, bjk -> bik', H, x_real)
    y = y + sigma_noise * (torch.randn_like(y) + 1j * torch.randn_like(y)) / np.sqrt(2)
    
    x_true_cuda = x_true.cuda()
    y_cuda = y.cuda()
    H_cuda = H.cuda()
    
    # Method 1: Least Squares
    x_pred_ls = least_squares_baseline(y_cuda, H_cuda)
    ber_ls = torch.mean(((x_pred_ls > 0.5) != x_true_cuda).float()).item()
    results['ls'].append(ber_ls)
    
    # Method 2: Regular Inference (with more steps for fair comparison)
    x_pred_reg = regular_inference(
        model=model,
        y_rx=y_cuda,
        h_channel=H_cuda,
        device='cuda',
        num_steps=15
    )
    ber_reg = torch.mean(((x_pred_reg > 0.5) != x_true_cuda).float()).item()
    results['regular'].append(ber_reg)
    
    # Method 3: Split Gibbs (with fixed moderate likelihood scale)
    # Use FIXED small likelihood_scale that doesn't vary with SNR
    # At very low SNR, likelihood is too weak anyway
    likelihood_scale = 0.01
    x_pred_sg = split_gibbs_inference(
        model=model,
        y_rx=y_cuda,
        h_channel=H_cuda,
        device='cuda',
        num_diffusion_steps=15,
        num_mh_steps=100,
        snr_db=snr_db,
        likelihood_scale=likelihood_scale
    )
    ber_sg = torch.mean(((x_pred_sg > 0.5) != x_true_cuda).float()).item()
    results['split_gibbs'].append(ber_sg)
    
    print(f"  LS BER:         {ber_ls:.6f}")
    print(f"  Regular BER:    {ber_reg:.6f}")
    print(f"  SplitGibbs BER: {ber_sg:.6f}")

# Summary statistics
print(f"\n{'='*60}")
print(f"SUMMARY (3 trials)")
print(f"{'='*60}")

ls_mean = np.mean(results['ls'])
ls_std = np.std(results['ls'])
reg_mean = np.mean(results['regular'])
reg_std = np.std(results['regular'])
sg_mean = np.mean(results['split_gibbs'])
sg_std = np.std(results['split_gibbs'])

print(f"Least Squares:  {ls_mean:.6f} ± {ls_std:.6f}")
print(f"Regular:        {reg_mean:.6f} ± {reg_std:.6f}")
print(f"SplitGibbs:     {sg_mean:.6f} ± {sg_std:.6f}")

print(f"\nImprovements:")
if ls_mean > 0:
    print(f"  SG vs LS:      {100 * (ls_mean - sg_mean) / ls_mean:+.1f}%")
print(f"  SG vs Regular: {100 * (reg_mean - sg_mean) / reg_mean:+.1f}%")

print(f"\n{'='*60}\n")
