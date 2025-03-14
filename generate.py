import torch
import torchvision.utils as vutils
import os
from generator import Generator
from PIL import Image, ImageEnhance

# Load Generator Model
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if model exists
MODEL_PATH = "generator_epoch_100.pth"
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file {MODEL_PATH} not found. Train the model first!")
    exit()

generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
generator.eval()

# Generate Images
num_images = 10
noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
generated_images = generator(noise)

# Create output directory
os.makedirs("generated_samples", exist_ok=True)

# Save and enhance images
for i, img in enumerate(generated_images):
    img_path = f"generated_samples/generated_slide_{i}.png"
    
    # Save original image
    vutils.save_image(img, img_path, normalize=True)
    print(f"✅ Generated slide saved: {img_path}")
    
    # Load image and enhance
    image = Image.open(img_path)
    
    # Apply sharpening
    sharp_enhancer = ImageEnhance.Sharpness(image)
    image = sharp_enhancer.enhance(2.0)  # Adjust sharpness factor

    # Apply contrast enhancement
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(1.5)  # Adjust contrast factor

    # Save enhanced image
    enhanced_path = f"generated_samples/enhanced_slide_{i}.png"
    image.save(enhanced_path)
    print(f"✨ Enhanced slide saved: {enhanced_path}")
