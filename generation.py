import torch
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
from DDPM import DiffusionModel
from ShowImages import showImages
from DataLoading import saveData, loadDataFromFolder
from torch.utils.data import DataLoader
import torch

# Function to generate fake images using the trained model
def generate_fake_images(model, num_images, latent_dim, device):
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        noise = generate_noise(num_images, latent_dim).to(device)
        fake_images = model(noise)
    return fake_images.detach().cpu().numpy()


# Function to generate noise
def generate_noise(batch_size, latent_dim):
    return torch.randn(batch_size, latent_dim)


# Function to train the diffusion model
def train_diffusion_model(model, dataloader, num_epochs, latent_dim, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            real_images = batch
            real_images = real_images.to(device)

            # Generate noise for the diffusion model
            noise = generate_noise(real_images.size(0), latent_dim).to(device)

            # Generate fake images
            fake_images = model(noise)

            # Update the generator
            loss = criterion(fake_images, real_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
        )

        # Plot some generated images every few epochs
        # if (epoch + 1) % 5 == 0:
        #     showImages(fake_images, f"generated_images_epoch_{epoch + 1}.png")


def generate(train_loader, num_generated_images: int, device):
    num_epochs = 100
    image_size = 256
    latent_dim = 100

    # Initialize the diffusion model
    model = DiffusionModel(image_size=image_size, hidden_dim=latent_dim)

    # Train the diffusion model
    train_diffusion_model(model, train_loader, num_epochs, latent_dim, device)

    # Generate 1000 fake images
    fake_images = generate_fake_images(model, num_generated_images, latent_dim, device)

    showImages(fake_images, "Final Result")
    return fake_images



def main():
    original_data_path = "C:/Projects/Big-Files/brain_tumor_dataset/train/original/"
    generated_data_path = "C:/Projects/Big-Files/brain_tumor_dataset/train/generated/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    loader: DataLoader = loadDataFromFolder(original_data_path + "no")
    fake_images = generate(loader, 1000, device)
    saveData(fake_images, generated_data_path + "no")

    loader = loadDataFromFolder(original_data_path + "yes")
    fake_images = generate(loader, 1000, device)
    saveData(fake_images, generated_data_path + "yes")


if __name__ == "__main__":
    main()