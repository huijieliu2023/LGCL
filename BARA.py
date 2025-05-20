
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class FeatureAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, bottleneck_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def BARA(primary, fallback, device='cpu', epochs=50, lr=1e-3):
    feat = torch.tensor(primary, dtype=torch.float32, device=device) \
           if not isinstance(primary, torch.Tensor) else primary.to(device).float()
    fb   = torch.tensor(fallback, dtype=torch.float32, device=device) \
           if not isinstance(fallback, torch.Tensor) else fallback.to(device).float()
    N, D = feat.shape

    ae = FeatureAutoencoder(input_dim=D).to(device)
    opt = optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ae.train()
    for _ in tqdm(range(epochs)):
        opt.zero_grad()
        recon = ae(feat)
        loss = loss_fn(recon, feat)
        loss.backward()
        opt.step()

    ae.eval()
    with torch.no_grad():
        recon_p = ae(feat)     
        recon_f = ae(fb)         

        err_p = F.mse_loss(recon_p, feat, reduction='none').mean(dim=1)  # (N,)
        err_f = F.mse_loss(recon_f, fb,   reduction='none').mean(dim=1)  # (N,)

        mask = err_f < err_p    # Tensor(bool) of shape (N,)

        corrected = feat.clone()
        corrected[mask] = fb[mask]


    return corrected
