import torch


def train_efficientnet(model, train_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            inputs2 = inputs.unsqueeze(1)
            outputs = model(inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, epoch_loss))


def evaluate_efficientnet(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs2 = inputs.unsqueeze(1)
            outputs = model(inputs2)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / total
    class_accuracy = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(len(class_correct))
    ]
    print(
        "Test Loss: {:.4f}, Test Accuracy: {:.4f}%".format(
            test_loss, 100 * test_accuracy
        )
    )
    print("Class Accuracy: ", class_accuracy)
    return test_loss, test_accuracy, class_accuracy
