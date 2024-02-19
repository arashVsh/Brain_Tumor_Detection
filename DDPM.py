
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, image_size, hidden_dim=64, num_layers=4):
        super(DiffusionModel, self).__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Generator layers
        generator_layers = []
        generator_layers.append(nn.Linear(hidden_dim, 512))
        generator_layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            generator_layers.append(nn.Linear(512, 512))
            generator_layers.append(nn.ReLU())
        generator_layers.append(nn.Linear(512, image_size * image_size))
        generator_layers.append(nn.Sigmoid())
        self.generator = nn.Sequential(*generator_layers)

    def forward(self, noise):
        fake_images = self.generator(noise)
        return fake_images.view(-1, 1, self.image_size, self.image_size)
