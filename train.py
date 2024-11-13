import json
import os
import logging
from time import time
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from src.data_loader import get_dataloaders
from src.unet import UNet2D
from src.evaluate import evaluate, visualize_predictions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set up the logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():

    # Hyperparameters
    INITIAL_LR = 1e-3
    BATCH_SIZE = 32
    EPOCHS = 10  # Increase epochs for early stopping
    PATIENCE = 3  # Early stopping patience
    logger.info("Hyperparameters:\n"
                f'Initial Learning Rate: {INITIAL_LR}\n'
                f'Batch Size: {BATCH_SIZE}\n'
                f'Number of Epochs: {EPOCHS}\n')
    
    # Define train-val split
    TRAIN_SPLIT = 0.8
    TEST_SPLIT = 1 - TRAIN_SPLIT
    VAL_SPLIT = 0.25
    logger.info(f'Train-val split = {TRAIN_SPLIT:.2f}-{VAL_SPLIT:.2f}')

    # Set the device we will use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using {"GPU" if torch.cuda.is_available() else "CPU"} for training...')

    # Prepare DataLoader
    logger.info('Loading the data...')
    image_dir = 'data/img'
    label_dir = 'data/label'
    trainDataLoader, valDataLoader, testDataLoader = get_dataloaders(image_dir, label_dir,
                                                                     test_split=TEST_SPLIT,
                                                                     val_split=VAL_SPLIT,
                                                                     batch_size=BATCH_SIZE)
    logger.info("Split the training data into train-val split")

    # Initialise the UNet model
    logger.info("Initialising the UNet model...")
    model = UNet2D(in_channels=3, num_classes=1).to(device)
    lossFn = nn.BCELoss()  # or Dice Loss
    opt = optim.Adam(model.parameters(), lr=INITIAL_LR)

    # Noting the starting time
    logger.info("Training the network...")
    startTime = time()

    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience_counter = 0

    # Training Loop
    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0.0

        # Progress bar for training loop
        trainLoader = tqdm(trainDataLoader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training", leave=False)
        for images, labels in trainLoader:

            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = lossFn(outputs, labels)

            # Backward pass and optimization
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(trainDataLoader)
        train_losses.append(avg_train_loss)

        # Evaluate on validation data
        val_loss, val_iou, val_f1 = evaluate(valDataLoader, model, lossFn, device)
        val_losses.append(val_loss)

        # Log epoch metrics
        logger.info(f"Epoch [{epoch + 1}/{EPOCHS}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, F1 Score: {val_f1:.4f}")

        # Store validation metrics
        val_accuracies.append(val_iou)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'output/temp_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            logger.info("Early stopping!")
            break

    logger.info(f"Total training time: {time() - startTime:.2f} seconds.")

    # After training, visualize
    logger.info("Visualizing some predictions...")
    visualize_predictions(testDataLoader, model, device)

    # Evaluate and visualize predictions on test data
    logger.info("Evaluating the model on test data...")
    test_loss, test_iou, test_f1 = evaluate(testDataLoader, model, lossFn, device)
    logger.info(f"Test Loss: {test_loss:.4f}, IoU: {test_iou:.4f}, F1 Score: {test_f1:.4f}")
    logger.info("Visualizing some predictions on test data...")
    visualize_predictions(testDataLoader, model, device)
    
    # Create output directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Save evaluation metrics
    metrics = {
        'validation_loss': val_loss,
        'iou': val_iou,
        'f1_score': val_f1
    }

    metrics_path = os.path.join(output_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            old_metrics = json.load(f)
        old_val_loss = old_metrics['validation_loss']
        if val_loss < old_val_loss:
            torch.save(model.state_dict(), os.path.join(output_dir, 'unet_model.pth'))
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            logger.info(f"Model improved. Saved new model to {output_dir}")

            # Plot and save training and validation losses
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
            plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.ylim(0, 1)
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.savefig('output/loss_plot.png')
            plt.show()

            # Plot and save validation IoU
            plt.figure(figsize=(10, 5))
            # plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Training IoU')
            plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Validation IoU')
            plt.xlabel('Epochs')
            plt.ylabel('IoU')
            plt.ylim(0, 1)
            plt.title('Validation IoU')
            plt.legend()
            plt.savefig('output/iou_plot.png')
            plt.show()
        else:
            logger.info("Model did not improve. Not saving.")
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, 'unet_model.pth'))
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        logger.info(f"Model saved to {output_dir}")

         # Plot and save training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
        plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('output/loss_plot.png')
        plt.show()

        # Plot and save validation IoU
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Training IoU')
        plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Validation IoU')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.ylim(0, 1)
        plt.title('Validation IoU')
        plt.legend()
        plt.savefig('output/iou_plot.png')
        plt.show()


if __name__ == '__main__':
    main()
