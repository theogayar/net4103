import torch
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import RandomLinkSplit
import networkx as nx
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import random


# 1. Define the GNN Encoder as per 4(e)
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def build_node_features(G, feature_attrs):
    """One-hot encode categorical node attributes for link prediction."""

    def normalize_feature_value(value):
        if value in (None, 0, "0", "unknown", "Unknown"):
            return "missing"
        return str(value)

    feature_value_maps = {}
    feature_offsets = {}
    total_feature_dim = 0

    for attr in feature_attrs:
        values = [
            normalize_feature_value(G.nodes[node_id].get(attr, None))
            for node_id in G.nodes
        ]
        unique_values = sorted(set(values))
        feature_value_maps[attr] = {
            value: idx for idx, value in enumerate(unique_values)
        }
        feature_offsets[attr] = total_feature_dim
        total_feature_dim += len(unique_values)

    node_ids = list(G.nodes)
    features = torch.zeros((len(node_ids), total_feature_dim), dtype=torch.float32)
    for row_index, node_id in enumerate(node_ids):
        for attr in feature_attrs:
            value = normalize_feature_value(G.nodes[node_id].get(attr, None))
            column_index = feature_offsets[attr] + feature_value_maps[attr][value]
            features[row_index, column_index] = 1.0

    return features


def evaluate_gnn(file_name, data_dir="data", k_list=[50, 100, 200, 400]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = nx.Graph(nx.read_gml(os.path.join(data_dir, file_name)))
    G = nx.convert_node_labels_to_integers(G)

    # 1. Integrate Node Features (Satisfying 4e requirement)
    # Use one-hot encodings for categorical student attributes.
    feature_attrs = [
        "student_fac",
        "gender",
        "year",
        "major_index",
        "second_major",
        "dorm",
    ]
    x = build_node_features(G, feature_attrs).to(device)

    data = from_networkx(G)
    data.x = x

    # 2. Split edges correctly
    split = RandomLinkSplit(
        num_val=0.05, num_test=0.1, is_undirected=True, add_negative_train_samples=False
    )
    train_data, val_data, test_data = split(data)
    train_data = train_data.to(device)
    test_data = test_data.to(device)

    # 3. Model Setup (GAE)
    model = GAE(GCNEncoder(data.num_features, 16)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 4. Training
    model.train()
    for epoch in tqdm(
        range(1, 151), desc=f"Training {file_name}", unit="epoch", leave=False
    ):
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.edge_index)
        loss.backward()
        optimizer.step()

    # 5. Prediction and Correct Evaluation
    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)
        scores = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()

    # Create a mask of edges used in training
    adj_train = np.zeros((len(G), len(G)))
    u_tr, v_tr = train_data.edge_index.cpu().numpy()
    adj_train[u_tr, v_tr] = 1

    candidate_mask = np.triu(np.ones(scores.shape), k=1).astype(bool)
    candidate_mask[adj_train > 0] = False  # Only block edges the GNN already knows

    candidate_scores = scores[candidate_mask]
    candidate_indices = np.argwhere(candidate_mask)
    sorted_idx = np.argsort(candidate_scores)[::-1]

    # Ground Truth: These are the edges removed by RandomLinkSplit
    pos_test_edge_index = test_data.edge_label_index[:, test_data.edge_label == 1]
    removed_edges = set(
        tuple(sorted((u, v))) for u, v in pos_test_edge_index.t().tolist()
    )

    results = []
    for k in k_list:
        top_k_indices = sorted_idx[:k]
        tp = 0
        for idx in top_k_indices:
            pair = tuple(sorted(candidate_indices[idx]))
            if pair in removed_edges:
                tp += 1
        results.append({"Metric": "GNN (GAE)", "k": k, "Precision@k": tp / k})

    return pd.DataFrame(results)


if __name__ == "__main__":
    # --- Execute on 15 Schools ---
    # all_files = [f for f in os.listdir("data") if f.endswith(".gml")]
    # selected_schools = random.sample(all_files, 15)

    # Here, running with the random selection is fine, and I never encountered the previous OOM error.
    # We still define a fixed set of schools to ensure reproducibility and reduce runtime for testing purposes.
    selected_schools = [
        "Caltech36.gml",
        "Reed98.gml",
        "Simmons81.gml",
        "Haverford76.gml",
        "Swarthmore42.gml",
        "Amherst41.gml",
        "Bowdoin47.gml",
        "Hamilton46.gml",
        "Trinity100.gml",
        "Williams40.gml",
        "Wellesley22.gml",
        "Vassar85.gml",
        "Smith60.gml",
        "Oberlin44.gml",
        "Wesleyan43.gml",
    ]

    gnn_results_list = []

    for school in selected_schools:
        print(f"Training GNN for {school}...")
        res = evaluate_gnn(school)
        gnn_results_list.append(res)

    master_gnn_df = pd.concat(gnn_results_list)
    print(master_gnn_df.groupby("k")["Precision@k"].mean())
