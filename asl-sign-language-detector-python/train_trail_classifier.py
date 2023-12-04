import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from train_EMNIST import CNN
from dataloader import TrailImageDataset


def get_trail_classifier():
    # Load final TRAIL classifier
    model = CNN()
    model.fc2 = nn.Linear(64, 3)
    state_dict = torch.load('asl-sign-language-detector-python/models/final_trail_model_2.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_EMNIST_classifier():
    # Load pretrained EMNIST classifier and replace output layer
    model = CNN()
    state_dict = torch.load('asl-sign-language-detector-python\models\final_trail_model_2.pth')
    model.load_state_dict(state_dict)
    model.fc2 = nn.Linear(64, 3)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TrailImageDataset(root='./images/train')
    test_dataset = TrailImageDataset(root='./images/test')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = get_EMNIST_classifier()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    best_model = None
    best_val_loss = 1000

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        eval_loss = 0.0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {running_loss/len(train_loader)}, Val loss: {eval_loss/len(test_loader)}")
        if eval_loss / len(test_loader) < best_val_loss:
            best_val_loss = eval_loss / len(test_loader)
            best_model = model
            print('New best model found. Saving checkpoint...')

    # Testing loop
    best_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    torch.save(best_model.state_dict(), 'asl-sign-languagae-detector/models/final_trail_model_2.pth')