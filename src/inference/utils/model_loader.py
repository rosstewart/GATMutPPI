import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
import glob
import numpy as np


class MutPred_PPI(nn.Module):
    """GAT_mut_processor — canonical hidden_dim=64 architecture."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1,
                 num_heads=4, mutation_diff_dim=1024):
        super(MutPred_PPI, self).__init__()

        self.mutation_diff_processor = nn.Sequential(
            nn.Linear(mutation_diff_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
        )

        self.complex_gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.complex_gat2 = GATConv(hidden_dim * num_heads, hidden_dim // 2, heads=1, concat=False)

        self.binding_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, output_dim),
        )

    def forward(self, x, edge_index, mutation_idx, mutation_site_diff):
        if mutation_site_diff.dim() == 1:
            mutation_site_diff = mutation_site_diff.unsqueeze(0)
        processed_mut_diff = self.mutation_diff_processor(mutation_site_diff)

        h = torch.relu(self.complex_gat1(x, edge_index))
        h = torch.relu(self.complex_gat2(h, edge_index))

        features_at_mutation = h[mutation_idx:mutation_idx + 1]
        combined = torch.cat([features_at_mutation, processed_mut_diff], dim=-1)
        return self.binding_predictor(combined)


def get_models(model_dir, device):
    input_dim = 1024

    models = []
    primary = f'{model_dir}/MutPred-PPI.pt'
    if glob.glob(primary):
        model_paths = [primary]
    else:
        # Ensemble directory (e.g. per-fold checkpoints): load every matching file.
        model_paths = glob.glob(f'{model_dir}/MutPred-PPI_*_megascale_all_*.pt')

    for model_path in model_paths:
        model = MutPred_PPI(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.eval()
        models.append(model)

    assert len(models) != 0, f"No model checkpoints found in {model_dir}"
    return models


# helper function to format input for ppi model
def format_model_input(embedding, edge_mat, device):
    features = torch.tensor(embedding, dtype=torch.float).to(device)
    edge_index = torch.tensor(edge_mat)
    edge_index, _ = dense_to_sparse(edge_index)
    edge_index = edge_index.to(device)
    return features, edge_index


def model_predict_subgraph(node_emb, edge_index_np, models, mut_local_idx, mutation_site_diff_np, device):
    """Run ensemble inference on a pre-built 2-hop subgraph.

    Args:
        node_emb:              (k, 1024) float32 numpy — subgraph node features
        edge_index_np:         (2, e) int32 numpy — COO edges in local coords
        models:                list of MutPred_PPI models
        mut_local_idx:         int — mutation site index within local nodes
        mutation_site_diff_np: (1024,) float32 numpy — scaled mutation diff
        device:                torch.device
    """
    try:
        x = torch.tensor(node_emb, dtype=torch.float).to(device)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)
        mut_diff_t = torch.tensor(mutation_site_diff_np, dtype=torch.float).to(device)

        if x.size(0) == 0 or edge_index.size(1) == 0:
            return None
        if edge_index.max() >= x.size(0):
            print(f"[ERROR] subgraph edge_index out of bounds: "
                  f"max={edge_index.max()}, nodes={x.size(0)}")
            return None

        preds = []
        for model in models:
            with torch.no_grad():
                out = model(x, edge_index, mut_local_idx, mut_diff_t)
                if out.size(0) == 0:
                    return None
                preds.append(torch.sigmoid(out).squeeze().cpu().numpy())

        return float(np.mean(preds))

    except RuntimeError as e:
        if "indexSelectLargeIndex" in str(e):
            print(f"[CUDA INDEX ERROR] {e}")
            return None
        raise


def model_predict(embedding, edge_mat, models, mutation_idx, mutation_site_diff, device):
    try:
        features, edge_index = format_model_input(embedding, edge_mat, device)
        mutation_site_diff = torch.tensor(mutation_site_diff, dtype=torch.float).to(device)

        if features is None or edge_index is None:
            return None
        if features.size(0) == 0 or edge_index.size(1) == 0:
            return None

        preds = []

        for i, model in enumerate(models):
            with torch.no_grad():
                if edge_index.max() >= features.size(0):
                    print(f"[ERROR] edge_index out of bounds: max={edge_index.max()}, features={features.size(0)}")
                    return None

                out = model(features, edge_index, mutation_idx, mutation_site_diff)
                if out.size(0) == 0:
                    return None
                pred = torch.sigmoid(out).squeeze().cpu().numpy()
                preds.append(pred)

        mean_pred = np.mean(np.array(preds), axis=0)
        return mean_pred

    except RuntimeError as e:
        if "indexSelectLargeIndex" in str(e):
            print(f"[CUDA INDEX ERROR] {e}")
            return None
        raise
