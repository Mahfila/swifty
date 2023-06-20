import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionNetwork(nn.Module):
    def __init__(self, embedding_size, hidden_units=512):
        super().__init__()
        self.hidden_dim = hidden_units
        self.emb_dim = embedding_size
        self.encoder = nn.LSTM(embedding_size, hidden_units, num_layers=1, bidirectional=True)
        self.fc1 = nn.Linear(hidden_units, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

    @staticmethod
    def attention_layer(encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        attention_wights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        alphas = F.softmax(attention_wights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1, 2), alphas.unsqueeze(2)).squeeze(2)
        return alphas, new_hidden

    def forward(self, x):
        output, (encoder_hidden, cell_state) = self.encoder(x)
        bidirectional_sum_initial = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        bidirectional_sum_initial = bidirectional_sum_initial.permute(1, 0, 2)
        bidirectional_sum = (encoder_hidden[-2, :, :] + encoder_hidden[-1, :, :]).unsqueeze(0)
        alphas, attn_out = self.attention_layer(bidirectional_sum_initial, bidirectional_sum)
        x = F.relu(self.fc1(attn_out))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
