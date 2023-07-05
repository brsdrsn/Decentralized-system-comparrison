import torch
import torch.nn as nn

# The recurrent neural network
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  # Add two batch dimensions of size 1
        hidden = self.init_hidden(1)  # Initialize hidden state with batch size 1

        out, hidden = self.rnn(x, hidden)

        out = out.squeeze(0).squeeze(0)  # Remove the extra batch dimensions
        out = self.fc(out)

        return out

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim)

