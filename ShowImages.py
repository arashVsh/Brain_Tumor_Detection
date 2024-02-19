import numpy as np
import matplotlib.pyplot as plt
import torch

def showImages(image_list, title):
    if type(image_list) is torch.Tensor:
        image_list = image_list.detach().cpu().numpy()
    if image_list.shape[1] == 1:
        # Reshape the array to (758, 128, 128)
        image_list = np.squeeze(image_list, axis=1)

    # Randomly select 25 indices
    random_indices = np.random.choice(image_list.shape[0], 25, replace=False)

    # Create a 5x5 subplot grid
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))

    for i, ax in enumerate(axes.flatten()):
        # Get a random index
        idx = random_indices[i]
        
        # Select the image from the array
        img = image_list[idx]  # Shape: (128, 128)
        
        # Display the image
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)  # Add title to the figure
    plt.show()


