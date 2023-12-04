from dataloader import BoundingBoxFeatures
from sequence_classifier import SequenceClassifier
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceClassifier(input_dim=2, hidden_dim=16, output_dim=3).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = BoundingBoxFeatures(split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_dataset = BoundingBoxFeatures(split='test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    best_avg_eval_loss = 1000000
    best_model = None
    for epoch in range(10):
        train_loss = 0
        model.train()
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            preds = model(inputs)
            loss = loss_fn(preds, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss

        eval_loss = 0
        model.eval()
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                preds = model(inputs)

                loss = loss_fn(preds, targets)
            eval_loss += loss


        avg_train_loss = train_loss / len(train_dataset)
        avg_eval_loss = eval_loss / len(test_dataset)
        print(f'Train loss at epoch {epoch}: {avg_train_loss}')
        print(f'Eval loss at epoch: {epoch}: {avg_eval_loss}')
        if avg_eval_loss < best_avg_eval_loss:
            best_avg_eval_loss = avg_eval_loss
            best_model = model
            print('New best model found. Saving checkpoint...')

    # Eval loop
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
    torch.save(model.state_dict(), 'models/lstm_model_2.pth')


if __name__ == '__main__':
    train()