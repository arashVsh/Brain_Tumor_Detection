import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader


def classify_ResNet(train_loader: DataLoader, test_loader: DataLoader):
    from CustomResNet import CustomResNetWithSelfAttention, trainResNet, evaluateResNet

    model = CustomResNetWithSelfAttention(
        num_classes=2
    )  # Assuming you have 2 classes for binary classification
    # print(model)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Assuming you have train_loader and test_loader already defined
    trainResNet(model, train_loader, criterion, optimizer, num_epochs=50)
    test_loss, test_acc, class_accuracy = evaluateResNet(model, test_loader, criterion)
    return test_loss, test_acc, class_accuracy


def classify_EfficientNet(train_loader: DataLoader, test_loader: DataLoader):
    from CustomEfficientNet import evaluate_efficientnet, train_efficientnet

    model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=2)

    # Modify the first convolutional layer to accept grayscale images
    model._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_efficientnet(model, train_loader, criterion, optimizer, num_epochs=50)
    test_loss, test_acc, class_accuracy = evaluate_efficientnet(
        model, test_loader, criterion
    )
    return test_loss, test_acc, class_accuracy
