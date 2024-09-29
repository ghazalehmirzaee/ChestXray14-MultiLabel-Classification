import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from data.augmentations import ACBA


def generate_augmented_dataset(original_data_dir, original_label_file, output_dir, target_samples_per_class=20000):
    print("Starting augmented dataset generation...")

    # Load original labels
    labels_df = pd.read_csv(original_label_file, sep=' ', header=None)
    image_files = labels_df.iloc[:, 0].values
    targets = labels_df.iloc[:, 1:].values.astype(np.float32)

    # Calculate class frequencies
    class_frequencies = targets.sum(axis=0)
    num_classes = targets.shape[1]

    # Initialize ACBA
    acba = ACBA()
    acba.initialize(targets)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create directories for each class
    class_dirs = []
    for i in range(num_classes):
        class_dir = os.path.join(output_dir, f"class_{i}")
        os.makedirs(class_dir, exist_ok=True)
        class_dirs.append(class_dir)

    # Generate augmented images
    new_image_files = []
    new_targets = []

    print("Generating augmented images...")
    for idx, (img_file, target) in enumerate(tqdm(zip(image_files, targets), total=len(image_files))):
        img_path = os.path.join(original_data_dir, img_file)
        image = Image.open(img_path).convert('RGB')

        # Determine how many times to augment this image
        augmentation_counts = []
        for i, label in enumerate(target):
            if label == 1:
                count = max(1, int(target_samples_per_class / class_frequencies[i]))
                augmentation_counts.append(count)

        max_augmentations = max(augmentation_counts) if augmentation_counts else 1

        for aug_idx in range(max_augmentations):
            # Apply ACBA
            augmented_image = acba.apply_augmentations(image, target)

            # Generate a new filename
            base_name, ext = os.path.splitext(img_file)
            new_filename = f"{base_name}_aug_{aug_idx}{ext}"

            # Save the augmented image in each relevant class directory
            for i, label in enumerate(target):
                if label == 1 and aug_idx < augmentation_counts[augmentation_counts.index(max(augmentation_counts))]:
                    save_path = os.path.join(class_dirs[i], new_filename)
                    augmented_image.save(save_path)

            new_image_files.append(new_filename)
            new_targets.append(target)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} original images")

    # Create new label file for augmented images
    print("Creating new label file...")
    new_label_file = os.path.join(output_dir, "augmented_labels.txt")
    with open(new_label_file, 'w') as f:
        for img_file, target in zip(new_image_files, new_targets):
            new_line = f"{img_file} " + " ".join(map(str, target.astype(int)))
            f.write(new_line + "\n")

    print(f"Augmented dataset generation complete. New label file: {new_label_file}")
    print(f"Augmented images are saved in class-specific subdirectories of: {output_dir}")

    return new_label_file, output_dir


if __name__ == "__main__":
    original_data_dir = "/NewRaidData/ghazal/data/ChestX-ray14/categorized_images/train"
    original_label_file = "/NewRaidData/ghazal/data/ChestX-ray14/labels/train_list.txt"
    output_dir = "/NewRaidData/ghazal/data/ChestX-ray14/augmented_dataset"

    generate_augmented_dataset(original_data_dir, original_label_file, output_dir)

