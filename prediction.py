import os
import argparse
from sklearn.metrics import f1_score, jaccard_score
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.unet import UNet2D

# Define transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_label = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def load_model(model_path, device):
    model = UNet2D(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model


def predict(model, image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = (output > 0.5).float().cpu().squeeze().numpy()
    return prediction


def visualize_prediction(image_path, label_path, prediction):
    image = Image.open(image_path)#.convert('RGB')
    #image = transform(image)

    label = Image.open(label_path).convert('L')
    label = transform_label(label).squeeze().numpy()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    # plt.imshow(image.permute(1, 2, 0))
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title('Original Label')
    plt.imshow(label, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Prediction')
    plt.imshow(prediction, cmap='gray')

    plt.suptitle(f'F1: {f1:.4f}, IoU: {iou:.4f}')
    plt.savefig('./output/prediction.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNet Prediction Script')
    parser.add_argument('--model', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--image', type=str, required=True, help='Path to the image to predict on')
    parser.add_argument('--target', type=str, required=True, help='Path to the image label')
    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    label_path = args.target
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, device)
    prediction = predict(model, image_path, transform, device)

    label = Image.open(label_path).convert('L')
    label = transform_label(label)
    label[label > 0.5] = 1.
    label[label <= 0.5] = 0.

    preds_np = prediction.flatten()
    labels_np = label.numpy().flatten()

    iou = jaccard_score(labels_np, preds_np, average='binary')
    f1 = f1_score(labels_np, preds_np, average='binary')

    print(f'IOU Score: {iou:.4f}')
    print(f'F1 score {f1:.4f}')

    visualize_prediction(image_path, label_path, prediction)
