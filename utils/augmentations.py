"""
Training augmentations for handwriting segmentation.
Includes geometric transforms, color jitter, shadow ramps,
sensor noise, and JPEG compression artifacts.
"""

import io
import random
import math
import numpy as np
import torch
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF


class TrainAugmentation:
    """
    Composite augmentation pipeline for image/mask pairs.
    Applies geometric + photometric transforms consistently.
    Final output: 768x768 random crop.
    """

    def __init__(self, crop_size=768):
        self.crop_size = crop_size

    def __call__(self, image: Image.Image, mask: Image.Image):
        """
        Args:
            image: PIL RGB image
            mask: PIL single-channel label map (0-3)
        Returns:
            image_tensor: [3, crop_size, crop_size] float32
            mask_tensor: [crop_size, crop_size] long
        """
        # --- Geometric transforms ---
        # Random scale 0.35-1.6x (CRITICAL: must go above 1.0)
        scale = random.uniform(0.35, 1.6)
        new_w = max(self.crop_size, int(image.width * scale))
        new_h = max(self.crop_size, int(image.height * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # Random rotation -15 to +15 degrees
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR,
                          fill=(255, 255, 255))
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST,
                         fill=0)

        # Random crop to crop_size x crop_size
        i, j, h, w = self._get_random_crop_params(image, self.crop_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # --- Photometric transforms (image only) ---
        # Color jitter
        image = self._color_jitter(image)

        # Shadow ramp
        if random.random() > 0.5:
            image = self._shadow_ramp(image)

        # Sensor noise (Gaussian)
        if random.random() > 0.5:
            image = self._sensor_noise(image)

        # JPEG compression artifacts
        if random.random() > 0.4:
            image = self._jpeg_artifacts(image)

        # Convert to tensors
        image_tensor = TF.to_tensor(image)  # [3, H, W] float [0,1]
        # Normalize: x/255 is already done by to_tensor, subtract 0.5
        image_tensor = image_tensor - 0.5

        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np)

        return image_tensor, mask_tensor

    def _get_random_crop_params(self, img, crop_size):
        w, h = img.size
        if h < crop_size or w < crop_size:
            # Pad if needed
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            img_padded = Image.new(img.mode, (w + pad_w, h + pad_h), (255, 255, 255) if img.mode == 'RGB' else 0)
            img_padded.paste(img, (0, 0))
            h, w = crop_size, crop_size
        top = random.randint(0, max(0, h - crop_size))
        left = random.randint(0, max(0, w - crop_size))
        return top, left, crop_size, crop_size

    def _color_jitter(self, image):
        """Apply brightness, contrast, saturation jitter."""
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)

        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)
        return image

    def _shadow_ramp(self, image):
        """Apply random gradient shadow overlay."""
        w, h = image.size
        arr = np.array(image, dtype=np.float32)

        # Random direction
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        intensity = random.uniform(0.1, 0.4)

        if direction == 'horizontal':
            ramp = np.linspace(1.0, 1.0 - intensity, w).reshape(1, w, 1)
        elif direction == 'vertical':
            ramp = np.linspace(1.0, 1.0 - intensity, h).reshape(h, 1, 1)
        else:
            ramp_h = np.linspace(1.0, 1.0 - intensity * 0.7, h).reshape(h, 1, 1)
            ramp_w = np.linspace(1.0, 1.0 - intensity * 0.7, w).reshape(1, w, 1)
            ramp = ramp_h * ramp_w

        if random.random() > 0.5:
            ramp = np.flip(ramp, axis=random.randint(0, len(ramp.shape) - 1)).copy()

        arr = arr * ramp
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _sensor_noise(self, image):
        """Add Gaussian sensor noise."""
        arr = np.array(image, dtype=np.float32)
        sigma = random.uniform(3, 12)
        noise = np.random.normal(0, sigma, arr.shape)
        arr = arr + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _jpeg_artifacts(self, image):
        """Save and reload with JPEG compression to add artifacts."""
        quality = random.randint(60, 95)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class ValAugmentation:
    """Validation: just center-crop and normalize, no augmentation."""

    def __init__(self, crop_size=768):
        self.crop_size = crop_size

    def __call__(self, image: Image.Image, mask: Image.Image):
        # Resize to fit crop if needed
        w, h = image.size
        if w < self.crop_size or h < self.crop_size:
            scale = max(self.crop_size / w, self.crop_size / h)
            new_w = int(w * scale) + 1
            new_h = int(h * scale) + 1
            image = image.resize((new_w, new_h), Image.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.NEAREST)
            w, h = new_w, new_h

        # Center crop
        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2
        image = TF.crop(image, top, left, self.crop_size, self.crop_size)
        mask = TF.crop(mask, top, left, self.crop_size, self.crop_size)

        image_tensor = TF.to_tensor(image) - 0.5
        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np)

        return image_tensor, mask_tensor
