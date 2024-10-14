import json
import os
import logging
from time import time
import torch
import torch.nn as nn
from sklearn.metrics import jaccard_score, f1_score
from torch import optim
from tqdm import tqdm
from src.data_loader import get_dataloaders
from src.unet import UNet2D
from src.evaluate import evaluate, visualize_predictions
import numpy as np

# Set up the logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Hyperparameters
    INITIAL_LR = 1e-3
    BATCH_SIZE = 8
    EPOCHS = 1
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
    # Calculate steps per epoch
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE
    # Initialise the UNet model
    logger.info("Initialising the UNet model...")
    model = UNet2D(in_channels=3, num_classes=1).to(device)
    lossFn = nn.BCELoss()  # or Dice Loss
    opt = optim.Adam(model.parameters(), lr=INITIAL_LR)
    # Initialise a dictionary to store training history
    H = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    # Noting the starting time
    logger.info("Training the network...")
    startTime = time()
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
        H['train_loss'].append(running_loss / len(trainDataLoader))
        logger.info(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(trainDataLoader):.4f}")
    print("Training finished.")
    logger.info(f"Total training time: {time() - startTime:.2f} seconds.")
    # After training, evaluate and visualize
    logger.info("Evaluating the model...")
    val_loss, val_iou, val_f1 = evaluate(valDataLoader, model, lossFn, device)
    logger.info(f"Validation Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, F1 Score: {val_f1:.4f}")
    logger.info("Visualizing some predictions...")
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
        else:
            logger.info("Model did not improve. Not saving.")
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, 'unet_model.pth'))
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        logger.info(f"Model saved to {output_dir}")


if __name__ == '__main__':
    main()
