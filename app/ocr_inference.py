import os
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class OCRMobileNetV2(nn.Module):
    def __init__(self, num_classes=26, dropout_rate=0.3):
        super().__init__()
        self.features = models.mobilenet_v2(pretrained=False).features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

print("✓ Model class defined")
#load the model weights 
def load_model(model_path):
    model = OCRMobileNetV2(num_classes=26)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model
#prrepcessing pipeline: real photo → MNIST-style → ImageNet-normalised tensor

def to_mnist_style(image_path: str):
    """
    Convert a real handwritten photo to MNIST-style:
      - Remove ruled lines
      - Isolate the ink stroke
      - Black background + white letter
      - Centered and padded like MNIST
    Returns both the final 28x28 numpy image and each intermediate stage.
    """
    stages = {}

    # ── 1. Load as BGR then convert to grayscale ──────────────────────────
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    stages['1_grayscale'] = gray.copy()

    # ── 2. Remove ruled lines using morphological operations ──────────────
    # Detect horizontal lines (كشكول lines are horizontal and thin)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines    = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
                                         horizontal_kernel, iterations=2)
    # Subtract the lines from the image (fill with white=255)
    no_lines = cv2.add(gray, detected_lines)
    no_lines = np.clip(no_lines, 0, 255).astype(np.uint8)
    stages['2_lines_removed'] = no_lines.copy()

    # ── 3. Isolate blue/dark ink via adaptive thresholding ────────────────
    # Adaptive threshold handles uneven lighting and varying ink intensity
    thresh = cv2.adaptiveThreshold(
        no_lines, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,   # INV → ink becomes white, background black
        blockSize=31,
        C=10
    )
    stages['3_thresholded'] = thresh.copy()

    # ── 4. Denoise — remove small specks left after thresholding ──────────
    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    denoised    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    stages['4_denoised'] = denoised.copy()

    # ── 5. Find bounding box of the letter and crop tightly ───────────────
    coords = cv2.findNonZero(denoised)
    if coords is None:
        # Nothing detected — return blank
        stages['5_cropped'] = np.zeros((28, 28), dtype=np.uint8)
        stages['6_final_28x28'] = np.zeros((28, 28), dtype=np.uint8)
        return stages

    x, y, w, h = cv2.boundingRect(coords)
    cropped = denoised[y:y+h, x:x+w]
    stages['5_cropped'] = cropped.copy()

    # ── 6. Resize to 20x20 (MNIST leaves a 4-pixel border on each side) ──
    letter_resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)

    # ── 7. Pad to 28x28 with black (center the letter like MNIST) ─────────
    final = np.zeros((28, 28), dtype=np.uint8)
    final[4:24, 4:24] = letter_resized
    stages['6_final_28x28'] = final.copy()

    return stages

def preprocess(image_path: str) -> torch.Tensor:
    """Full pipeline: real photo → MNIST-style → ImageNet-normalised tensor."""
    stages = to_mnist_style(image_path)
    img_28 = stages['6_final_28x28']           # uint8 [0, 255], black bg

    # Upscale to 224x224 (what MobileNetV2 expects)
    img_224 = cv2.resize(img_28, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Stack to 3 channels
    img_rgb = np.stack([img_224] * 3, axis=-1).astype(np.uint8)

    # Convert to PIL and apply ImageNet normalisation
    pil_img = Image.fromarray(img_rgb, mode='RGB')
    tensor  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])(pil_img).unsqueeze(0)

    return tensor, stages
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
def predict(model, tensor):
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()
    confidence, pred_idx = probs.max(dim=0)
    return CLASSES[pred_idx.item()], confidence.item() * 100
