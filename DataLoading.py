from torchvision import transforms
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from CustomDataset import MyCustomDataset

ROOT_TRAIN_PATH = "C:/Projects/Big-Files/brain_tumor_dataset/train/"
ROOT_TEST_PATH = "C:/Projects/Big-Files/brain_tumor_dataset/test/"


def mergeDatasets(datasets: list[Dataset], stringLabels: list[str], training: bool):
    images = []
    labels: list[int] = []

    for index in range(len(datasets)):
        if stringLabels[index].lower() == "y" or stringLabels[index].lower() == "yes":
            for element in datasets[index]:
                images.append(element[0])
                labels.append(1)
        else:
            for element in datasets[index]:
                image = element[0]
                # if image.shape[0] > 3:
                #     print('yes')
                #     image = image.unsqueeze(0)
                # else:
                #     print('no')
                images.append(image)
                labels.append(0)

    combined_dataset = MyCustomDataset(images, labels)
    combined_loader = DataLoader(combined_dataset, batch_size=16, shuffle=training)
    return combined_loader


def load_and_partition_samples(folder_path):
    # Define transformations (e.g., converting images to tensors)
    transform = transforms.Compose(
        [transforms.Resize((160, 160)), transforms.Grayscale(), transforms.ToTensor()]
    )

    dataset = ImageFolderDataset(folder_path, transform=transform)
    partition_index = len(dataset) // 4

    subset1_indices = range(0, partition_index)
    subset1 = Subset(dataset, subset1_indices)

    subset2_indices = range(partition_index, 2 * partition_index)
    subset2 = Subset(dataset, subset2_indices)

    subset3_indices = range(2 * partition_index, 3 * partition_index + 1)
    subset3 = Subset(dataset, subset3_indices)

    subset4_indices = range(3 * partition_index + 1, len(dataset))
    subset4 = Subset(dataset, subset4_indices)

    return subset1, subset2, subset3, subset4


# Custom dataset class to load images from a folder
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = [
            os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
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
