# import matplotlib.pyplot as plt
from ShowImages import showImages
from CustomDataset import MyCustomDataset
from torchvision import transforms


def generate(num_images: int, train_loader):

    transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), shear=0.1, fill=(0,)
            ),  # Random affine transformation
            transforms.ColorJitter(
                brightness=(0.3, 1.0)
            ),  # Random brightness adjustment
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomVerticalFlip(),  # Random vertical flip
            transforms.RandomPerspective(
                distortion_scale=0.1, p=0.5, fill=(0,)
            ),  # Random perspective transformation
            transforms.ToTensor(),  # Convert PIL Image to tensor
        ]
    )

    generated_images = []
    # Apply transformations and convert to numpy array
    while len(generated_images) < num_images:
        for batch in train_loader:
            for image in batch[0]:
                to_pil = transforms.ToPILImage()
                pil_image = to_pil(image)
                transformed_image = transform(pil_image)
                transformed_image = transformed_image.numpy()  # Convert to numpy array
                generated_images.append(transformed_image)
                if len(generated_images) == num_images:
                    break
            if len(generated_images) == num_images:
                break

    # Convert the list of numpy arrays to a single numpy array
    showImages(generated_images, "Generated Images")
    generated_dataset = MyCustomDataset(generated_images)
    return generated_dataset
