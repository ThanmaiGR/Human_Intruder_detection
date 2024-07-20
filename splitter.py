import os
import shutil
import random


def create_split_folders(base_dir, split_names):
    for split in split_names:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)


def copy_files(file_list, source_dir, dest_dir):
    for file in file_list:
        shutil.copy(os.path.join(source_dir, file), os.path.join(dest_dir, file))


def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # Get list of all images
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    # Calculate split sizes
    total_images = len(images)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)

    # Split the images
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    # Ensure output directories exist
    create_split_folders(output_dir, ['train', 'val', 'test'])

    # Copy image and corresponding label files to respective directories
    for split, split_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
        split_images_dir = os.path.join(output_dir, split, 'images')
        split_labels_dir = os.path.join(output_dir, split, 'labels')

        copy_files(split_images, images_dir, split_images_dir)
        for image in split_images:
            label_file = image.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(split_labels_dir, label_file))

    return train_images, val_images, test_images


def create_data_yaml(output_dir, nc, names):
    data_yaml_content = f"""
train: {os.path.join(output_dir, 'train/images')}
val: {os.path.join(output_dir, 'val/images')}
test: {os.path.join(output_dir, 'test/images')}

nc: {nc}
names: {names}
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(data_yaml_content)


# Define paths
base_dir = 'img for human intrusion/img for human intrusion'
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')
output_dir = 'img for human intrusion'

# Split the dataset
train_images, val_images, test_images = split_dataset(images_dir, labels_dir, output_dir)

# Define class names (adjust this list based on your dataset)
class_names = ["intruder", "student", "faculty", ]  # Add your class names here

# Create the data.yaml file
create_data_yaml(output_dir, nc=len(class_names), names=class_names)
