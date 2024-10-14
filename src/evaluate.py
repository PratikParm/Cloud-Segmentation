import torch
from sklearn.metrics import jaccard_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate(loader, model, criterion, device):
    logger = logging.getLogger(__name__)
    model = model.to(device)
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Apply threshold to ensure predictions are binary
            preds = (outputs > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            logger.info(f"Predictions: {preds}")
            logger.info(f"Labels: {labels}")

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        iou = jaccard_score(all_labels.flatten(), all_preds.flatten(), average='binary')
        f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='binary')
    except ValueError as e:
        logger.error(f"Error calculating metrics: {e}")
        return val_loss / len(loader), 0, 0

    avg_loss = val_loss / len(loader)
    return avg_loss, iou, f1


def visualize_predictions(loader, model, device, num_images=5):
    model = model.to(device)
    model.eval()
    images_shown = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()

            for i in range(min(num_images, images.size(0))):
                plt.figure(figsize=(12, 4))

                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())

                plt.subplot(1, 3, 1)
                plt.title('Input Image')
                plt.imshow(img)

                plt.subplot(1, 3, 2)
                plt.title('Ground Truth')
                plt.imshow(labels[i].cpu().squeeze(), cmap='gray')

                plt.subplot(1, 3, 3)
                plt.title('Prediction')
                plt.imshow(preds[i].cpu().squeeze(), cmap='gray')

                plt.show()

                images_shown += 1
                if images_shown >= num_images:
                    return


if __name__ == "__main__":
    from unet import UNet2D

    def test_evaluate():
        from torch.utils.data import DataLoader, TensorDataset

        # Create synthetic data
        images = torch.rand(3000, 3, 128, 128)
        labels = (torch.rand(3000, 1, 128, 128) > 0.5).float()
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=2)

        model = UNet2D(in_channels=3, num_classes=1)
        model.eval()

        criterion = nn.BCELoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        avg_loss, iou, f1 = evaluate(loader, model, criterion, device)
        print(f"Test Loss: {avg_loss:.4f}, IoU: {iou:.4f}, F1 Score: {f1:.4f}")


    test_evaluate()
