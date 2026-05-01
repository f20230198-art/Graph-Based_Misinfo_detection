import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool


class HybridGATWithAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        motif_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.motif_mlp = nn.Sequential(
            nn.Linear(motif_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, edge_index, batch, motif_feat):
        h = self.input_norm(x)
        h = self.gat1(h, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.gat2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        h_graph = global_mean_pool(h, batch)

        m = self.motif_mlp(motif_feat)

        cat = torch.cat([h_graph, m], dim=1)
        alpha = self.gate(cat)
        fused = alpha * h_graph + (1.0 - alpha) * m

        return self.classifier(fused)


class GATGraphClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, heads: int = 4, dropout: float = 0.4):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        x = self.input_norm(x)
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GraphSAGEClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        x = self.input_norm(x)
        x = F.relu(self.bn1(self.sage1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.sage2(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.classifier(x)
