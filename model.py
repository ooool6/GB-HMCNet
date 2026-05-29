import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, LayerNorm
from torch.nn import Linear, Dropout


class GCN_8_plus(torch.nn.Module):
    def __init__(self, num_features, num_classes, initdim=16, inithead=16, edge_dim=12):
        super(GCN_8_plus, self).__init__()

        self.conv1 = GATConv(num_features, initdim, heads=inithead, edge_dim=edge_dim)
        self.BatchNorm1 = BatchNorm(initdim * inithead)
        self.conv_linear1 = torch.nn.Linear(initdim * inithead, initdim)
        self.BatchNorml1 = BatchNorm(initdim)

        self.conv2 = GATConv(initdim, initdim * 2, heads=int(inithead / 2), edge_dim=edge_dim)
        self.BatchNorm2 = BatchNorm(initdim * inithead)
        self.conv_linear2 = torch.nn.Linear(initdim * inithead, initdim * 2)
        self.BatchNorml2 = BatchNorm(initdim * 2)

        self.conv3 = GATConv(initdim * 2, initdim * 4, heads=int(inithead / 4), edge_dim=edge_dim)
        self.BatchNorm3 = BatchNorm(initdim * inithead)

        self.linear = torch.nn.Linear(initdim * inithead, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # block 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.BatchNorm1(x)
        x = F.relu(x)
        x = self.conv_linear1(x)
        x = self.BatchNorml1(x)
        x = F.relu(x)
        # block2

        x = self.conv2(x, edge_index, edge_attr)
        x = self.BatchNorm2(x)
        x = F.relu(x)
        x = self.conv_linear2(x)
        x = self.BatchNorml2(x)
        x = F.relu(x)

        # block 3
        x = self.conv3(x, edge_index, edge_attr)
        x = self.BatchNorm3(x)
        x = F.relu(x)

        x = self.linear(x)

        return x

class GCN_block(torch.nn.Module):
    def __init__(self, input_dims, output_dims, head_nums, do_linear=True, linear_outdims=None):
        super(GCN_block, self).__init__()

        self.do_linear = do_linear
        self.conv0 = GATConv(input_dims, output_dims, heads=head_nums, edge_dim=3)
        self.BN0 = BatchNorm(output_dims * head_nums)
        self.relu = torch.nn.ReLU()
        if self.do_linear:
            self.linear = torch.nn.Linear(output_dims * head_nums, linear_outdims)
            self.BN1 = BatchNorm(linear_outdims)

    def forward(self, x, adj, edge_attr):

        x = self.conv0(x, adj, edge_attr=edge_attr)
        x = self.BN0(x)
        x = self.relu(x)

        if self.do_linear:
            x = self.linear(x)

            x = self.BN1(x)
            x = self.relu(x)

        return x

if __name__ == '__main__':
    torch.manual_seed(0)
    num_features = 25
    num_classes = 17
    edge_dim = 3
    model = GCN_8_plus(num_features=num_features, num_classes=num_classes, initdim=64, inithead=8, edge_dim=edge_dim)

    num_nodes = 100
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_attr = torch.randn(num_nodes * 2, edge_dim)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    output = model(data)
    print(f"Output shape: {output.shape}")