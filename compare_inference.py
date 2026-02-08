"""
Compare Regular Inference vs Split Gibbs Sampling
Tests both methods at different SNR levels and generates comparison plots
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from MIMO_score_based_diffusion import TransformerScoreModel, simulate_rayleigh_mimo
from inference_methods import regular_inference, split_gibbs_inference, least_squares_baseline


def compute_ber(x_est, x_true):
    """Compute Bit Error Rate"""
    errors = (x_est != x_true).sum().item()
    total = x_true.numel()
    return errors / total, errors, total


def compare_methods(snr_values=[-10, -5, 0, 5, 10, 15, 20], 
                   num_trials=5,
                   batch_size=4,
                   device='cuda'):
    """
    Compare regular vs split gibbs inference across SNR levels
    
    Args:
        snr_values: List of SNR in dB to test
        num_trials: Number of trials per SNR
        batch_size: Batch size for each trial
        device: 'cuda' or 'cpu'
    
    Returns:
        results: Dictionary with BER results
    """
    
    print(f"\n{'='*70}")
    print(f"COMPARING INFERENCE METHODS (Broad SNR Range)")
    print(f"Testing SNRs: {snr_values} dB")
    print(f"Methods: Least Squares | Regular (5 steps) | Split Gibbs (15 steps + 100 MH)")
    print(f"{'='*70}\n")
    
    # Load model
    print("[Loading model...]")
    model = TransformerScoreModel(embedding_dim=256, num_layers=6)
    checkpoint = torch.load('mimo_sgdd_model.pth')
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("[Model loaded]\n")
    
    # Storage for results
    results = {
        'snr': snr_values,
        'ls': {'ber': [], 'ber_std': []},
        'regular': {'ber': [], 'ber_std': []},
        'split_gibbs': {'ber': [], 'ber_std': []},
    }
    
    # Test at each SNR
    for snr in snr_values:
        print(f"Testing at SNR = {snr:+.1f} dB...")
        
        regular_bers = []
        splitgibbs_bers = []
        ls_bers = []
        
        for trial in range(num_trials):
            # Generate test data
            b_true = torch.randint(0, 2, (batch_size, 2, 780)).float().to(device)
            x_tx, y_rx, h_channel = simulate_rayleigh_mimo(b_true, snr_db=snr, frame_size=52)
            b_flat = b_true.view(batch_size, -1).long()
            
            # ========== LEAST SQUARES BASELINE ==========
            x_ls = least_squares_baseline(y_rx, h_channel)
            ber_ls, _, _ = compute_ber(x_ls, b_flat)
            ls_bers.append(ber_ls)
            
            # ========== REGULAR INFERENCE ==========
            x_regular = regular_inference(model, y_rx, h_channel, device=device, num_steps=15)
            ber_regular, _, _ = compute_ber(x_regular, b_flat)
            regular_bers.append(ber_regular)
            
            # ========== SPLIT GIBBS INFERENCE ==========
            # Adaptive likelihood scaling: grows with SNR
            # At low SNR: scale=0.01 (protect from noise)
            # At high SNR: scale grows to 1.0 (amplify clean signal)
            likelihood_scale = max(0.01, 10.0 ** (snr / 10.0) * 0.01)
            
            x_splitgibbs = split_gibbs_inference(
                model, y_rx, h_channel, device=device,
                num_diffusion_steps=15, num_mh_steps=100, snr_db=snr,
                likelihood_scale=likelihood_scale
            )
            ber_splitgibbs, _, _ = compute_ber(x_splitgibbs, b_flat)
            splitgibbs_bers.append(ber_splitgibbs)
            
            print(f"  Trial {trial+1}/{num_trials}: LS={ber_ls:.4f}, Regular={ber_regular:.4f}, SplitGibbs={ber_splitgibbs:.4f}")
        
        # Compute statistics
        ls_mean = np.mean(ls_bers)
        ls_std = np.std(ls_bers)
        regular_mean = np.mean(regular_bers)
        regular_std = np.std(regular_bers)
        splitgibbs_mean = np.mean(splitgibbs_bers)
        splitgibbs_std = np.std(splitgibbs_bers)
        
        results['ls']['ber'].append(ls_mean)
        results['ls']['ber_std'].append(ls_std)
        results['regular']['ber'].append(regular_mean)
        results['regular']['ber_std'].append(regular_std)
        results['split_gibbs']['ber'].append(splitgibbs_mean)
        results['split_gibbs']['ber_std'].append(splitgibbs_std)
        
        improvement_vs_ls = 100 * (ls_mean - splitgibbs_mean) / ls_mean if ls_mean > 0 else 0
        improvement_vs_regular = 100 * (regular_mean - splitgibbs_mean) / regular_mean if regular_mean > 0 else 0
        print(f"  → LS: {ls_mean:.4f}±{ls_std:.4f}")
        print(f"  → Regular: {regular_mean:.4f}±{regular_std:.4f}")
        print(f"  → SplitGibbs: {splitgibbs_mean:.4f}±{splitgibbs_std:.4f}")
        print(f"  → Improvement vs LS: {improvement_vs_ls:.1f}%")
        print(f"  → Improvement vs Regular: {improvement_vs_regular:.1f}%\n")
    
    return results


def plot_results(results, save_path='inference_comparison.png'):
    """Plot BER comparison"""
    snr = results['snr']
    ls_ber = results['ls']['ber']
    ls_std = results['ls']['ber_std']
    regular_ber = results['regular']['ber']
    regular_std = results['regular']['ber_std']
    splitgibbs_ber = results['split_gibbs']['ber']
    splitgibbs_std = results['split_gibbs']['ber_std']
    
    plt.figure(figsize=(11, 7))
    
    # Plot with error bars
    plt.errorbar(snr, ls_ber, yerr=ls_std, marker='^', label='Least Squares', linewidth=2, markersize=8)
    plt.errorbar(snr, regular_ber, yerr=regular_std, marker='o', label='Regular Iterative', linewidth=2, markersize=8)
    plt.errorbar(snr, splitgibbs_ber, yerr=splitgibbs_std, marker='s', label='Split Gibbs', linewidth=2, markersize=8)
    
    plt.semilogy()
    plt.xlabel('SNR (dB)', fontsize=13)
    plt.ylabel('Bit Error Rate (BER)', fontsize=13)
    plt.title('MIMO Detection: LS vs Regular vs Split Gibbs (Broad SNR Range)', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {save_path}")
    plt.close()


def print_summary(results):
    """Print summary table"""
    print(f"\n{'='*90}")
    print(f"SUMMARY RESULTS (Broad SNR Range)")
    print(f"{'='*90}")
    print(f"{'SNR':<8} {'LS BER':<15} {'Regular BER':<15} {'SplitGibbs BER':<15} {'SG vs LS':<12}")
    print(f"{'-'*90}")
    
    for i, snr in enumerate(results['snr']):
        ls_ber = results['ls']['ber'][i]
        reg_ber = results['regular']['ber'][i]
        sg_ber = results['split_gibbs']['ber'][i]
        improvement_vs_ls = 100 * (ls_ber - sg_ber) / ls_ber if ls_ber > 0 else 0
        
        print(f"{snr:+.0f}dB   {ls_ber:.6f}        {reg_ber:.6f}        {sg_ber:.6f}        {improvement_vs_ls:+.1f}%")
    
    print(f"{'='*90}\n")


if __name__ == "__main__":
    # Run comparison across broader SNR range
    results = compare_methods(
        snr_values=[-10, -5, 0, 5, 10, 15, 20],
        num_trials=5,
        batch_size=4,
        device='cuda'
    )
    
    # Print summary
    print_summary(results)
    
    # Plot
    plot_results(results)
    
    print("✓ Comparison complete!")
