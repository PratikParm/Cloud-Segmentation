import os
import logging
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# Set up the logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CloudDataset(Dataset):
    def __init__(self, data_files, label_files, transform_image=None, transform_label=None):
        super(CloudDataset, self).__init__()

        self.transform_image = transform_image
        self.transform_label = transform_label
        self.data_files = data_files
        self.label_files = label_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        img_path = self.data_files[idx]
        label_path = self.label_files[idx]

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error opening image at {img_path}: {e}")
            raise

        try:
            label = Image.open(label_path).convert('L')
        except Exception as e:
            logger.error(f"Error opening label at {label_path}: {e}")
            raise

        if self.transform_image:
            img = self.transform_image(img)
        if self.transform_label:
            label = self.transform_label(label)

        return img, label


def get_dataloaders(image_dir, label_dir, val_split, test_split, batch_size):
    # Get list of files
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')])

    logger.info(f'Total data files: {len(image_files)}')

    # Split the dataset
    train_images, test_images, train_labels, test_labels = train_test_split(image_files[:1000], label_files[:1000], test_size=test_split, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=val_split, random_state=42)  # 0.25 * 0.8 = 0.2

    # Define transformations
    transform_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_label = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Create datasets
    train_dataset = CloudDataset(train_images, train_labels, transform_image=transform_img, transform_label=transform_label)
    val_dataset = CloudDataset(val_images, val_labels, transform_image=transform_img, transform_label=transform_label)
    test_dataset = CloudDataset(test_images, test_labels, transform_image=transform_img, transform_label=transform_label)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info('Data loaders created')

    return train_loader, val_loader, test_loader
