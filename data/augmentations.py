import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates

class ACBA:
    def __init__(self, alpha=0.5, beta=0.7, af_max=60):
        self.alpha = alpha
        self.beta = beta
        self.af_max = af_max
        self.augmentation_factors = None
        self.co_occurrence_matrix = None

    def initialize(self, labels):
        class_frequencies = {i: labels[:, i].sum() for i in range(labels.shape[1])}
        self.augmentation_factors = self.calculate_augmentation_factor(class_frequencies)
        self.co_occurrence_matrix = self.calculate_co_occurrence_matrix(labels)

    def calculate_augmentation_factor(self, class_frequencies):
        max_freq = max(class_frequencies.values())
        return {cls: min(1 + self.alpha * (max_freq / freq - 1), self.af_max)
                for cls, freq in class_frequencies.items()}

    def calculate_co_occurrence_matrix(self, labels):
        return np.dot(labels.T, labels)

    def should_augment(self, labels):
        return np.random.rand() < self.beta and np.sum(labels) > 0

    def calculate_co_occurrence_adjustment(self, labels):
        adjustment = 0
        total_co_occurrences = np.sum(self.co_occurrence_matrix)
        for i, label in enumerate(labels):
            if label == 1:
                adjustment += np.sum(self.co_occurrence_matrix[i]) / total_co_occurrences
        return adjustment

    def apply_augmentations(self, image, labels):
        if not self.should_augment(labels):
            return image

        co_occurrence_adjustment = self.calculate_co_occurrence_adjustment(labels)
        augmentation_multiplier = max(1, max([self.augmentation_factors[i] for i, label in enumerate(labels) if label == 1]) * (1 + co_occurrence_adjustment))

        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            self.apply_clahe,
            self.apply_gaussian_blur,
            self.apply_elastic_transform,
            self.apply_random_erasing,
            self.apply_cutout,
            self.apply_solarization
        ]

        num_augmentations = min(int(augmentation_multiplier), len(augmentations))
        selected_augmentations = np.random.choice(augmentations, num_augmentations, replace=False)

        for aug in selected_augmentations:
            image = aug(image)

        return image

    @staticmethod
    def apply_clahe(image):
        image = np.array(image)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final)

    @staticmethod
    def apply_gaussian_blur(image):
        return image.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0, 1)))

    @staticmethod
    def apply_elastic_transform(image):
        image = np.array(image)
        shape = image.shape
        if len(shape) == 2:
            image = np.expand_dims(image, axis=2)
            shape = image.shape
        alpha = shape[1] * np.random.uniform(1, 3)
        sigma = shape[1] * 0.07
        random_state = np.random.RandomState(None)
        dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        distorted_image = np.zeros_like(image)
        for c in range(shape[2]):
            distorted_image[:, :, c] = map_coordinates(image[:, :, c], indices, order=1, mode='reflect').reshape(shape[:2])
        return Image.fromarray(distorted_image.squeeze())

    @staticmethod
    def apply_random_erasing(image):
        if np.random.rand() < 0.5:
            return image
        w, h = image.size
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        area = w * h
        target_area = np.random.uniform(0.02, 0.4) * area
        aspect_ratio = np.random.uniform(0.3, 1/0.3)
        eh = int(np.sqrt(target_area * aspect_ratio))
        ew = int(np.sqrt(target_area / aspect_ratio))
        if x + ew > w:
            ew = w - x
        if y + eh > h:
            eh = h - y
        color = np.random.randint(0, 255, (3,))
        image = np.array(image)
        image[y:y+eh, x:x+ew] = color
        return Image.fromarray(image)

    @staticmethod
    def apply_cutout(image):
        if np.random.rand() < 0.5:
            return image
        w, h = image.size
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        cut_w = np.random.randint(w // 4, w // 2)
        cut_h = np.random.randint(h // 4, h // 2)
        image = np.array(image)
        image[y:y+cut_h, x:x+cut_w] = 0
        return Image.fromarray(image)

    @staticmethod
    def apply_solarization(image):
        img_array = np.array(image)
        solarized = np.where(img_array < 128, img_array, 255 - img_array)
        return Image.fromarray(solarized.astype(np.uint8))

def get_transform(is_train):
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

