import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from generator import Generator
from discriminator import Discriminator

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (same as training)
latent_dim = 100
image_size = 64
batch_size = 16

# Data Transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load Dataset
dataset = datasets.ImageFolder(root="storeimg", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load Models
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Load Trained Model
MODEL_PATH = "generator_epoch_100.pth"  # Change if needed
generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
generator.eval()
discriminator.eval()

# Loss Function
criterion = torch.nn.BCELoss()

# Recover Loss Values
d_losses = []
g_losses = []

with torch.no_grad():
    for images, _ in dataloader:
        images = images.to(device)
        batch_size = images.size(0)

        # Compute Discriminator Loss
        real_outputs = discriminator(images)
        real_loss = criterion(real_outputs, torch.ones_like(real_outputs))

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        fake_outputs = discriminator(fake_images)
        fake_loss = criterion(fake_outputs, torch.zeros_like(fake_outputs))

        d_loss = real_loss + fake_loss
        d_losses.append(d_loss.item())

        # Compute Generator Loss
        g_loss = criterion(fake_outputs, torch.ones_like(fake_outputs))
        g_losses.append(g_loss.item())

# Save Losses for Future Use
np.savez("loss_data_recovered.npz", d_losses=np.array(d_losses), g_losses=np.array(g_losses))
print("✅ Recovered loss data saved as loss_data_recovered.npz")

# Print Average Loss
print(f"Recovered Loss - Avg D Loss: {np.mean(d_losses):.4f}, Avg G Loss: {np.mean(g_losses):.4f}")
