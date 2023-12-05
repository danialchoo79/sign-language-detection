from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=128, output_dim=26):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    model = LinearClassifier()
    print('Hi, you ran the wrong script :P')