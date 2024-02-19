from torch.utils.data import DataLoader


def detailsDisplayer(dataLoader: DataLoader):
    print("Batch Size:", dataLoader.batch_size)
    print("Number of Batches:", len(dataLoader))
    print("Total Samples:", len(dataLoader.dataset))

    class_samples_count = [0] * 2
    for _, label in dataLoader.dataset:
        class_samples_count[label] += 1

    print("Samples per class:")
    for i, count in enumerate(class_samples_count):
        print(f"Class {i}: {count} samples")
    print()

    # Iterate over DataLoader to access batches of images and labels
    for batch_images, batch_labels in dataLoader:
        # Process each batch as needed
        print("Batch images shape:", batch_images.shape)
        print("Batch labels:", batch_labels)
        break  # Print only the first batch for demonstration
