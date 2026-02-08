"""
Inference Methods for MIMO Detection
Two implementations: Regular iterative denoising vs Split Gibbs sampling
"""
import torch
import numpy as np
from MIMO_score_based_diffusion import TransformerScoreModel, simulate_rayleigh_mimo


class MIMOLikelihood:
    """Compute log p(y|x) for MIMO channel"""
    
    def __init__(self, h_matrix, y_received, snr_db=10.0):
        self.h = h_matrix
        self.y = y_received
        snr_lin = 10 ** (snr_db / 10.0)
        self.sigma_sq = 1.0 / (2 * snr_lin)
        self.device = h_matrix.device
    
    def forward(self, x_bits):
        """Compute reconstruction error ||y - H(x)||²"""
        batch_size = x_bits.shape[0]
        # Convert bits to BPSK
        x_tx = (2 * x_bits.float() - 1).to(torch.complex64)
        x_tx = x_tx.view(batch_size, 2, 780)
        
        # Compute y_hat = H @ x_tx
        y_hat = torch.einsum('bijk, bjk -> bik', self.h, x_tx)
        
        # Compute error per batch element (not averaged)
        error = torch.sum((self.y - y_hat).abs() ** 2, dim=(1, 2))
        return error
    
    def log_likelihood(self, x_bits):
        """Log probability p(y|x) = -error / sigma²"""
        error = self.forward(x_bits)
        return -error / self.sigma_sq


def least_squares_baseline(y, h):
    """
    Least squares MIMO detection baseline
    x̂ = (H^H H)^-1 H^H y
    """
    h_mats = h.permute(0, 3, 1, 2)  # (batch, frames, 2, 2)
    y_vecs = y.transpose(1, 2).unsqueeze(-1)  # (batch, frames, 2, 1)
    
    # Solve least squares per frame
    x_hat = torch.linalg.lstsq(h_mats, y_vecs).solution.squeeze(-1)  # (batch, frames, 2)
    x_hat = x_hat.transpose(1, 2).reshape(x_hat.shape[0], -1)  # (batch, 1560)
    
    # Convert to bits: threshold at 0
    x_bits = (x_hat.real > 0).long()
    return x_bits


def regular_inference(model, y_rx, h_channel, device='cuda', num_steps=5):
    """
    Regular iterative denoising inference
    Simply applies diffusion model without likelihood refinement
    
    Args:
        model: Trained TransformerScoreModel
        y_rx: Received signal (batch, 2, T)
        h_channel: Channel matrix (batch, 2, 2, frames)
        device: 'cuda' or 'cpu'
        num_steps: Number of denoising iterations
    
    Returns:
        x_estimate: Estimated bits (batch, 1560)
    """
    batch_size = y_rx.shape[0]
    y_rx = y_rx.to(device)
    h_channel = h_channel.to(device)
    
    # Initialize with random bits
    x_noisy = torch.randint(0, 2, (batch_size, 1560)).float().to(device)
    
    # Iterative denoising
    for step in reversed(range(1, num_steps + 1)):
        sigma = torch.full((batch_size,), step / num_steps).to(device)
        
        with torch.no_grad():
            logits = model(x_noisy.long(), sigma, y_rx, h_channel)
            x_noisy = torch.argmax(logits, dim=-1).float()
    
    return x_noisy.long()


def split_gibbs_inference(model, y_rx, h_channel, device='cuda', 
                          num_diffusion_steps=10, num_mh_steps=50, snr_db=10.0,
                          likelihood_scale=1.0):
    """
    Split Gibbs sampling: alternates prior (diffusion) and likelihood (MH) steps
    
    Args:
        model: Trained TransformerScoreModel
        y_rx: Received signal (batch, 2, T)
        h_channel: Channel matrix (batch, 2, 2, frames)
        device: 'cuda' or 'cpu'
        num_diffusion_steps: T (diffusion annealing steps)
        num_mh_steps: K (Metropolis-Hastings iterations per step)
        snr_db: Signal-to-noise ratio in dB
        likelihood_scale: Scale factor for likelihood (higher = stronger conditioning)
    
    Returns:
        x_estimate: Estimated bits (batch, 1560)
    """
    batch_size = y_rx.shape[0]
    y_rx = y_rx.to(device)
    h_channel = h_channel.to(device)
    
    # Initialize likelihood operator
    likelihood = MIMOLikelihood(h_channel, y_rx, snr_db=snr_db)
    
    # Annealing schedule
    time_steps = torch.linspace(1, 0, num_diffusion_steps + 1)[:-1]
    
    # Better initialization: start with diffusion denoising at high noise level
    sigma_init = torch.full((batch_size,), 1.0, device=device)
    with torch.no_grad():
        logits = model(torch.randint(0, 2, (batch_size, 1560)).to(device).long(), 
                      sigma_init, y_rx, h_channel)
        x_t = torch.argmax(logits, dim=-1).float()
    
    seq_len = x_t.shape[1]
    
    for t in time_steps:
        # ========== PRIOR STEP ==========
        sigma_batch = torch.full((batch_size,), t.item(), device=device)
        with torch.no_grad():
            logits = model(x_t.long(), sigma_batch, y_rx, h_channel)
            x_0_hat = torch.argmax(logits, dim=-1).float()
        
        # ========== LIKELIHOOD STEP ==========
        log_p_current = likelihood.log_likelihood(x_0_hat) * likelihood_scale
        x_refined = x_0_hat.clone()
        
        for mh_iter in range(num_mh_steps):
            # Propose: flip only 1 bit (less aggressive)
            flip_indices = torch.randint(0, seq_len, (batch_size, 1), device=device)
            
            x_proposal = x_refined.clone()
            x_proposal[torch.arange(batch_size), flip_indices.squeeze()] = \
                1 - x_proposal[torch.arange(batch_size), flip_indices.squeeze()]
            
            # Evaluate
            log_p_proposal = likelihood.log_likelihood(x_proposal) * likelihood_scale
            log_ratio = log_p_proposal - log_p_current
            accept_prob = torch.exp(torch.clamp(log_ratio, max=0))
            
            # Accept/reject
            u = torch.rand(batch_size, device=device)
            accept_mask = u < accept_prob
            
            x_refined[accept_mask] = x_proposal[accept_mask]
            log_p_current[accept_mask] = log_p_proposal[accept_mask]
        
        x_t = x_refined
    
    return x_t.long()
