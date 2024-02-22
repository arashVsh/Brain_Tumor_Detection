import os
import shutil
import random


# Function to split images into train and test sets
def split_dataset(source_folder, train_dest_folder, test_dest_folder, split_ratio):
    # Create train and test directories for each class
    train_class_folder = os.path.join(train_dest_folder, os.path.basename(source_folder))
    test_class_folder = os.path.join(test_dest_folder, os.path.basename(source_folder))

    # Delete the old train and test folders if they exist.
    if os.path.exists(train_class_folder):
        shutil.rmtree(train_class_folder)
    if os.path.exists(test_class_folder):
        shutil.rmtree(test_class_folder)

    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(test_class_folder, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(source_folder)]
    
    # Calculate the number of images for train and test
    num_images = len(image_files)
    num_train = int(num_images * split_ratio)
    # num_test = num_images - num_train
    
    # Shuffle the list of image files
    random.shuffle(image_files)
    
    # Copy images to train and test folders
    for i, image_file in enumerate(image_files):
        source_path = os.path.join(source_folder, image_file)
        if i < num_train:
            dest_path = os.path.join(train_class_folder, image_file)
        else:
            dest_path = os.path.join(test_class_folder, image_file)
        shutil.copy(source_path, dest_path)


def main():
    # Path to the main dataset folder
    dataset_folder = "C:/Projects/Big-Files/brain_tumor_dataset"

    # Create train and test directories
    train_folder = os.path.join(dataset_folder, "train/original")
    test_folder = os.path.join(dataset_folder, "test")

    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Split "yes" images
    split_dataset(os.path.join(dataset_folder, "yes"), train_folder, test_folder, split_ratio=0.6)

    # Split "no" images
    split_dataset(os.path.join(dataset_folder, "no"), train_folder, test_folder, split_ratio=0.6)

    print("Dataset split completed successfully.")


if __name__ == '__main__':
    main()