import torch
import torch.nn as nn
from Vit import VisionTransformer
import matplotlib.pyplot as plt


def evaluate(train_loader, test_loader, device):
    # Hyperparameters
    learning_rate = 1e-4
    num_epochs = 30

    train_losses = []
    test_losses = []

    model = VisionTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0  # Accumulator for total training loss in the epoch
        total_train_samples = 0  # Accumulator for total number of training samples in the epoch

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)  # Accumulate loss per batch
            total_train_samples += images.size(0)  # Accumulate number of samples per batch

        # Compute average training loss for the epoch
        epoch_train_loss = total_train_loss / total_train_samples
        train_losses.append(epoch_train_loss)  # Store total training loss for the epoch

        # Evaluation loop
        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0
            total_test_samples = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item() * images.size(0)
                total_test_samples += images.size(0)

            # Compute average test loss for the epoch
            epoch_test_loss = total_test_loss / total_test_samples
            test_losses.append(epoch_test_loss)  # Store total test loss for the epoch

        # Print loss after each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}")

    epchos = range(num_epochs)
    plt.plot(epchos, train_losses, label="Training Loss")
    plt.plot(epchos, test_losses, "-.", label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Diagram")
    plt.show()

    # Test the model
    model.eval()
    with torch.no_grad():
        correct_total = 0
        total_total = 0
        correct_class = [0] * num_epochs  # List to store correct predictions for each class
        total_class = [0] * num_epochs  # List to store total samples for each class

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Total accuracy
            total_total += labels.size(0)
            correct_total += (predicted == labels).sum().item()

            # Class-wise accuracy
            for label, pred in zip(labels, predicted):
                if label == pred:
                    correct_class[label] += 1
                total_class[label] += 1

        # Compute total accuracy
        accuracy_total = round(100 * correct_total / total_total)
        print(f"Total Test Accuracy: {accuracy_total}%")

        # Compute class-wise accuracy
        for label in range(2):
            accuracy_class = (
                round(100 * correct_class[label] / total_class[label])
                if total_class[label] > 0
                else 0
            )
            print(f"Class {label} Test Accuracy: {accuracy_class}%")
