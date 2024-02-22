import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CustomResNet import CustomResNetWithSelfAttention, trainModel, evaluateModel


def classify(train_loader: DataLoader, test_loader: DataLoader):
    model = CustomResNetWithSelfAttention(
        num_classes=2
    )  # Assuming you have 2 classes for binary classification
    # print(model)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Assuming you have train_loader and test_loader already defined
    trainModel(model, train_loader, criterion, optimizer, num_epochs=40)
    test_loss, test_acc, class_accuracy = evaluateModel(model, test_loader, criterion)
    return test_loss, test_acc, class_accuracy
