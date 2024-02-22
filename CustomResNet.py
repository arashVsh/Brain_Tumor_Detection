import torch.nn as nn
import torch.nn as nn

# import torchvision.models as models
import torch
import matplotlib.pyplot as plt


# class CustomResNet(nn.Module):
#     def __init__(self):
#         super(CustomResNet, self).__init__()
#         self.resnet = models.resnet18(pretrained=True)
#         # Replace the first convolutional layer to accept 1 channel input
#         self.resnet.conv1 = nn.Conv2d(
#             1, 64, kernel_size=7, stride=2, padding=3, bias=False
#         )

#         # Modify the last layer for binary classification
#         num_ftrs = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_ftrs, 2)

#     def forward(self, x):
#         return self.resnet(x)

#     def evaluateModel(self, test_loader):
#         self.eval()
#         correct = [0, 0]  # Number of correct predictions for each class
#         total = [0, 0]  # Total number of samples for each class

#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 outputs = self(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 for pred, label in zip(predicted, labels):
#                     if pred == label:
#                         correct[label] += 1
#                     total[label] += 1

#         accuracy_class0 = correct[0] / total[0] if total[0] != 0 else 0
#         accuracy_class1 = correct[1] / total[1] if total[1] != 0 else 0

#         print("Accuracy on class 0: {:.2f}%".format(100 * accuracy_class0))
#         print("Accuracy on class 1: {:.2f}%".format(100 * accuracy_class1))

#         overall_accuracy = sum(correct) / sum(total) if sum(total) != 0 else 0
#         print("Overall accuracy on test set: {:.2f}%".format(100 * overall_accuracy))

#     # Assuming you have a function to train your model
#     def trainModel(self, train_loader, criterion, optimizer, num_epochs):
#         losses = []  # To store the training losses for each epoch

#         for epoch in range(num_epochs):
#             self.train()
#             running_loss = 0.0
#             for inputs, labels in train_loader:
#                 optimizer.zero_grad()
#                 outputs = self(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item() * inputs.size(0)
#             epoch_loss = running_loss / len(train_loader.dataset)
#             losses.append(epoch_loss)
#             print(
#                 "Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, epoch_loss)
#             )

#         # Plotting the training loss
#         plt.plot(losses, label="Training Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.title("Training Loss")
#         plt.legend()
#         plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        proj_query = (
            self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)
        out = self.gamma * out + x
        return out


class ResidualBlockWithSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockWithSelfAttention, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sa = SelfAttention(out_channels)

        # Adding a 1x1 convolutional layer for adjusting dimensions in the residual connection
        self.conv_res = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        residual = self.conv_res(x)  # Adjusting dimensions of the residual connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sa(out)
        out += residual  # Adding the adjusted residual connection
        out = self.relu(out)
        return out


class CustomResNetWithSelfAttention(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNetWithSelfAttention, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResidualBlockWithSelfAttention, 64, 3)
        self.layer2 = self.make_layer(ResidualBlockWithSelfAttention, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResidualBlockWithSelfAttention, 256, 6, stride=2)
        self.layer4 = self.make_layer(ResidualBlockWithSelfAttention, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, channels))
        self.in_channels = channels
        for _ in range(1, num_blocks):
            layers.append(block(channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def trainModel(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs=10,
    device="cuda",
    save_path="best_model.pth",
):
    model.to(device)
    best_loss = float("inf")  # Initialize with a very large value
    best_model_state = None

    losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            inputs2 = inputs.unsqueeze(1)
            outputs = model(inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        acc = 100 * correct / total

        # Check if the current model has the best training loss so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = model.state_dict()  # (The lowest training loss)
        losses.append(epoch_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {acc:.2f}%"
        )

    # Save the best model's state
    torch.save(best_model_state, save_path)
    print(f"Best model saved with training loss: {best_loss:.4f}")
    # Plotting the training loss
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()


def evaluateModel(model, test_loader, criterion, device="cuda"):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    class_correct = [0] * 2
    class_total = [0] * 2

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs2 = inputs.unsqueeze(1)
            outputs = model(inputs2)
            loss = criterion(
                outputs, labels
            )  # Calculate the loss using the provided criterion
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    class_accuracy = [
        100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(2)
    ]

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    print("Class-wise Accuracy:", class_accuracy)
    return test_loss, test_acc, class_accuracy
