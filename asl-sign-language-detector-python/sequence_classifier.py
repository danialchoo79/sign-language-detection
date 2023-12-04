import torch
from torch import nn


class SequenceClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=3):
        super(SequenceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape

        x = x.view(batch_size * seq_len, -1)
        x = self.fc1(x)
        x = x.view(batch_size, seq_len, -1)

        x, _ = self.lstm(x)

        x = self.fc2(x[:, -1, :])

        return x
    

# --------- TEST ----------
if __name__ == "__main__":
    batch_size = 32
    seq_len = 60
    input_dim = 4

    input_tensor = torch.randn(batch_size, seq_len, input_dim)
    print(f'Input shape: {input_tensor.shape}')

    model = SequenceClassifier(input_dim=input_dim, hidden_dim=16, output_dim=3)

    output = model(input_tensor)

    print("Output shape:", output.shape) # Should be (batch_size, output_dim)