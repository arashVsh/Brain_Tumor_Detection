from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from MergeData import mergeDatasets

ROOT_TRAIN_PATH = "C:/Projects/Big-Files/brain_tumor_dataset/train/"
ROOT_TEST_PATH = "C:/Projects/Big-Files/brain_tumor_dataset/test/"


def loadTest():
    yes_loader: DataLoader = loadDataFromFolder(
        folder_path=ROOT_TEST_PATH + "yes"
    )
    no_loader: DataLoader = loadDataFromFolder(
        folder_path=ROOT_TEST_PATH + "no"
    )

    total_train_loader = mergeDatasets(
        [yes_loader.dataset, no_loader.dataset],
        ["y", "n"],
        training=False,
    )
    return total_train_loader


def loadOnlyOriginalTrain():
    original_yes_loader: DataLoader = loadDataFromFolder(
        folder_path=ROOT_TRAIN_PATH + "original/yes"
    )
    original_no_loader: DataLoader = loadDataFromFolder(
        folder_path=ROOT_TRAIN_PATH + "original/no"
    )

    total_train_loader = mergeDatasets(
        [original_yes_loader.dataset, original_no_loader.dataset],
        ["y", "n"],
        training=True,
    )
    return total_train_loader


def loadOnlyGeneratedTrain():
    generated_yes_loader: DataLoader = loadDataFromFolder(
        folder_path=ROOT_TRAIN_PATH + "generated/yes"
    )
    generated_no_loader: DataLoader = loadDataFromFolder(
        folder_path=ROOT_TRAIN_PATH + "generated/no"
    )
    total_train_loader = mergeDatasets(
        [generated_yes_loader.dataset, generated_no_loader.dataset],
        ["y", "n"],
        training=True,
    )
    return total_train_loader


def loadOriginalAndGeneratedTrain():
    original_yes_loader: DataLoader = loadDataFromFolder(
        folder_path=ROOT_TRAIN_PATH + "original/yes"
    )
    original_no_loader: DataLoader = loadDataFromFolder(
        folder_path=ROOT_TRAIN_PATH + "original/no"
    )
    generated_yes_loader: DataLoader = loadDataFromFolder(
        folder_path=ROOT_TRAIN_PATH + "generated/yes"
    )
    generated_no_loader: DataLoader = loadDataFromFolder(
        folder_path=ROOT_TRAIN_PATH + "generated/no"
    )
    total_train_loader = mergeDatasets(
        [
            original_yes_loader.dataset,
            original_no_loader.dataset,
            generated_yes_loader.dataset,
            generated_no_loader.dataset,
        ],
        ["y", "n", "y", "n"],
        training=True,
    )
    return total_train_loader


def saveData(numpy_array, folder_path):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        # If the folder exists, delete its contents
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)

    # Iterate over the images in the NumPy array
    for i, image_array in enumerate(numpy_array):
        # Convert the NumPy array to PIL Image
        image = Image.fromarray(
            np.uint8(image_array.squeeze() * 255), mode="L"
        )  # Assuming pixel values are in [0, 1] range

        # Save the image to the folder in JPEG format
        image_path = os.path.join(folder_path, f"image_{i}.jpg")
        image.save(image_path)

    print(f"Saved {len(numpy_array)} images to {folder_path}")


def loadDataFromFolder(folder_path):
    # Define transformations (e.g., converting images to tensors)
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize images to 256x256
            transforms.Grayscale(),  # Convert images to grayscale
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ]
    )

    dataset = ImageFolderDataset(folder_path, transform=transform)

    # Create DataLoader with shuffling enabled
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Iterate over DataLoader to access batches of shuffled images
    # for batch in dataloader:
    #     # Process each batch as needed
    #     print(batch.shape)  # Example: printing the shape of the batch
    return dataloader


# Custom dataset class to load images from a folder
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = [
            os.path.join(folder_path, filename)
            for filename in os.listdir(folder_path)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image
