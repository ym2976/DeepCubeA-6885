import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, out_dim: int):
        super().__init__()
        self.one_hot_depth = one_hot_depth
        input_channels = one_hot_depth if one_hot_depth > 0 else 1
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256 * state_dim, out_dim)

    def forward(self, states_nnet):
        # One-hot encoding if necessary
        if self.one_hot_depth > 0:
            x = F.one_hot(states_nnet.long(), self.one_hot_depth).permute(0, 2, 1).float()  # (batch, depth, state_dim)
        else:
            x = states_nnet.unsqueeze(1).float()  # (batch, 1, state_dim)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
class RNNModel(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.one_hot_depth = one_hot_depth
        input_dim = one_hot_depth if one_hot_depth > 0 else 1
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, states_nnet):
        # One-hot encoding if necessary
        if self.one_hot_depth > 0:
            x = F.one_hot(states_nnet.long(), self.one_hot_depth).float()  # (batch, state_dim, depth)
        else:
            x = states_nnet.unsqueeze(2).float()  # (batch, state_dim, 1)

        x, _ = self.rnn(x)  # (batch, state_dim, hidden_dim)
        x = x[:, -1, :]  # Use the last time step
        x = self.fc(x)
        return x

class ResnetModel(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # first two hidden layers
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, states_nnet):
        x = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        x = self.fc_out(x)
        return x
