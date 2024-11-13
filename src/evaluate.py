import torch
from sklearn.metrics import jaccard_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch import nn
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate(loader, model, criterion, device):
    logger = logging.getLogger(__name__)
    model = model.to(device)
    model.eval()
    val_loss = 0

    # Variables to store cumulative metrics
    total_iou = 0
    total_f1 = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Apply threshold to ensure predictions are binary
            preds = (outputs > 0.5).float()

            # Convert to numpy arrays for metric calculations
            preds_np = preds.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()

            try:
                # Incrementally calculate IoU and F1 scores
                iou = jaccard_score(labels_np, preds_np, average='binary')
                f1 = f1_score(labels_np, preds_np, average='binary')

                total_iou += iou * len(labels_np)  # Weighted by number of samples
                total_f1 += f1 * len(labels_np)
                total_samples += len(labels_np)

            except ValueError as e:
                logger.error(f"Error calculating metrics: {e}")
                continue

            torch.cuda.empty_cache()

    # Compute average loss, IoU, and F1
    avg_loss = val_loss / len(loader)
    avg_iou = total_iou / total_samples
    avg_f1 = total_f1 / total_samples

    return avg_loss, avg_iou, avg_f1



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
                # img = (img - img.min()) / (img.max() - img.min())

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
