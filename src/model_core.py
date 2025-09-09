# src/model_core.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from preprocessing import PROCESSED_MATRIX_PATH, PROCESSED_DIR

LATENT_EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "latent_embeddings.parquet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# VAE Model

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2_mu = nn.Linear(64, latent_dim)
        self.fc2_logvar = nn.Linear(64, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar



# Loss function

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kld


# Training function

def train_vae(data, input_dim, latent_dim=4, epochs=100, batch_size=32, lr=1e-3):
    model = VAE(input_dim, latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.tensor(data.values, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataset):.6f}")
    return model


# Generate latent embeddings

def generate_latent_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data.values, dtype=torch.float32).to(DEVICE)
        mu, logvar = model.encode(x)
        embeddings = mu.cpu().numpy()
        df_embeddings = pd.DataFrame(embeddings, index=data.index, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
        df_embeddings.to_parquet(LATENT_EMBEDDINGS_PATH)
        print(f"Latent embeddings saved to {LATENT_EMBEDDINGS_PATH}")
        return df_embeddings



# Run pipeline

if __name__ == "__main__":
    # Load processed matrix
    df = pd.read_parquet(PROCESSED_MATRIX_PATH)
    input_dim = df.shape[1]
    print(f"Training VAE on {input_dim} assets over {df.shape[0]} days...")

    vae_model = train_vae(df, input_dim, latent_dim=4, epochs=100, batch_size=32, lr=1e-3)
    embeddings = generate_latent_embeddings(vae_model, df)
    print(embeddings.head())
