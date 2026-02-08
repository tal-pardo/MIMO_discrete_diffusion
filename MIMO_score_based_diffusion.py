import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from torch.amp import autocast, GradScaler
import os

# --- 1. GOOGLE DRIVE MOUNTING ---
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    SAVE_PATH = '/content/drive/MyDrive/mimo_sgdd_model.pth'
except ImportError:
    SAVE_PATH = 'mimo_sgdd_model.pth'

# --- 2. ARCHITECTURE ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.pe[:, :x.size(1), :]

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, sigma):
        return self.mlp(sigma.unsqueeze(-1) if sigma.dim()==1 else sigma)

class TransformerScoreModel(nn.Module):
    def __init__(self, vocab_size=2, seq_len=1560, embedding_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=seq_len)
        self.time_embed = TimeEmbedding(embedding_dim)
        self.mimo_conditioner = nn.Sequential(
            nn.Linear(12, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x_t, sigma, y_cond, h_cond):
        batch_size = x_t.size(0)
        x_embed = self.token_embed(x_t.long())
        y_feat = torch.stack([y_cond.real, y_cond.imag], dim=-1).transpose(1, 2).reshape(batch_size, -1, 4)
        h_feat = torch.stack([h_cond.real, h_cond.imag], dim=-1).permute(0, 3, 1, 2, 4).reshape(batch_size, -1, 8)
        mimo_feats = self.mimo_conditioner(torch.cat([y_feat, h_feat], dim=-1)).repeat(1, 2, 1)
        x_input = x_embed + mimo_feats + self.pos_encoding(x_embed) + self.time_embed(sigma).unsqueeze(1)
        return self.output_proj(self.transformer(x_input))

# --- 3. PHYSICS & BASELINES ---
def simulate_rayleigh_mimo(b_bits, snr_db, frame_size=52):
    device = b_bits.device
    b, n_ant, n_sym = b_bits.shape
    x_tx = (2 * b_bits - 1).to(torch.complex64)
    num_frames = math.ceil(n_sym / frame_size)
    H_frames = (torch.randn(b, 2, 2, num_frames, device=device) + 1j*torch.randn(b, 2, 2, num_frames, device=device)) / math.sqrt(2)
    H_full = torch.repeat_interleave(H_frames, frame_size, dim=-1)[..., :n_sym]
    y_rx = torch.einsum('bijk, bjk -> bik', H_full, x_tx)
    snr_lin = 10**(snr_db/10.0); noise_std = math.sqrt(1.0/(2*snr_lin))
    y_rx += noise_std * (torch.randn_like(y_rx) + 1j*torch.randn_like(y_rx))
    return x_tx, y_rx, H_full

def least_squares_baseline(y, h):
    h_mats, y_vecs = h.permute(0, 3, 1, 2), y.transpose(1, 2).unsqueeze(-1)
    x_hat = torch.linalg.lstsq(h_mats, y_vecs).solution.squeeze(-1)
    return (x_hat.real > 0).float().transpose(1, 2), x_hat

# --- 4. TRAINING & VALIDATION ---
def run_experiment():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerScoreModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler(DEVICE.type)
    val_epochs, val_hist = [], {-5: {'sg':[], 'ls':[]}, 0: {'sg':[], 'ls':[]}, 10: {'sg':[], 'ls':[]}}
    epoch = 0

    while True:
        epoch += 1
        model.train(); optimizer.zero_grad()
        b_clean = torch.randint(0, 2, (16, 2, 780)).float().to(DEVICE)
        x_tx, y, h = simulate_rayleigh_mimo(b_clean, 15.0, frame_size=52)
        x0, sigma = b_clean.view(16, -1).long(), torch.rand(16).to(DEVICE)
        mask = (torch.rand_like(x0.float()) < sigma.unsqueeze(-1)).long()
        xt = (x0 * (1 - mask) + (1 - x0) * mask)

        with autocast(device_type=DEVICE.type):
            logits = model(xt, sigma, y, h)
            loss = F.cross_entropy(logits.reshape(-1, 2), x0.view(-1))

        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

        if epoch % 50 == 0:
            model.eval()
            val_epochs.append(epoch)
            with torch.no_grad():
                for snr in [-5, 0, 10]:
                    bv = torch.randint(0, 2, (8, 2, 780)).float().to(DEVICE)
                    _, yv, hv = simulate_rayleigh_mimo(bv, snr, frame_size=52)
                    xt_eval = torch.randint(0, 2, (8, 1560)).to(DEVICE)
                    for step in reversed(range(1, 6)):
                        xt_eval = torch.argmax(model(xt_eval, torch.full((8,), step/5).to(DEVICE), yv, hv), dim=-1)
                    val_hist[snr]['sg'].append((xt_eval != bv.view(8, -1)).sum().item()/bv.numel())
                    pls_bits, _ = least_squares_baseline(yv, hv)
                    val_hist[snr]['ls'].append((pls_bits != bv).sum().item()/bv.numel())

            clear_output(wait=True)
            print("="*60 + "\nSGDD TRAINING STATUS\n" + "="*60)
            print(f"Epoch: {epoch} | Current Loss: {loss.item():.4f}")
            print(f"Sample X (bits 0-5): {x_tx[0, 0, :5].real.cpu().numpy()}")
            print(f"Sample Y (complex): {y[0, 0, :2].cpu().numpy()}")
            print(f"Save Location: {SAVE_PATH}")
            print("-" * 60)
            print("Training Constraint: Loss < 0.05")
            print("Channel Characteristic: Rayleigh (52-symbol frame)")
            print("="*60)

            plt.figure(figsize=(9, 4.5))
            for snr, color in zip([-5, 0, 10], ['blue', 'orange', 'green']):
                plt.semilogy(val_epochs, val_hist[snr]['sg'], '-o', color=color, label=f'SGDD {snr}dB')
                plt.semilogy(val_epochs, val_hist[snr]['ls'], '--', color=color, alpha=0.3, label=f'LS {snr}dB')
            plt.title(f"Dynamic Training | Loss: {loss.item():.4f}"); plt.ylabel("BER"); plt.legend(); plt.grid(True, which='both'); plt.show()

            if loss.item() < 0.05:
                # --- SAVE AND RELOAD VERIFICATION ---
                print(f"\n[SAVE] Target reached. Saving weights...")
                torch.save({'model_state': model.state_dict(), 'loss': loss.item()}, SAVE_PATH)

                print(f"[VERIFY] Testing reload from Google Drive...")
                checkpoint = torch.load(SAVE_PATH)
                test_model = TransformerScoreModel().to(DEVICE)
                test_model.load_state_dict(checkpoint['model_state'])
                test_model.eval()
                print(f"[SUCCESS] Weights verified. Final reloaded loss: {checkpoint['loss']:.4f}")
                break

if __name__ == "__main__":
    run_experiment()