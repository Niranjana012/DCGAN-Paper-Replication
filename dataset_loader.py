import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SlideDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Traverse subdirectories and collect all JPG files
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(".jpg"):
                    self.image_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

# Define transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def get_dataloader(root_dir, batch_size=16, shuffle=True):
    dataset = SlideDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
