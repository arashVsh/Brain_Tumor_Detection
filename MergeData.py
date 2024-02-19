from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch


class CustomMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def mergeDatasets(datasets: list[Dataset], stringLabels: list[str], training: bool):
    images = []
    labels: list[int] = []

    for index in range(len(datasets)):
        if stringLabels[index] == 'y':
            for image in datasets[index]:
                images.append(image)
                labels.append(1)
        else:
            for image in datasets[index]:
                images.append(image)
                labels.append(0)

    combined_dataset = CustomMNISTDataset(images, labels)
    combined_loader = DataLoader(combined_dataset, batch_size=16, shuffle=training)

    return combined_loader
