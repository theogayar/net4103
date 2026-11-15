import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # Propagation rule: H = σ(Â X W)
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


def normalize_adj(adj):
    # Renormalization Trick: Â = D^-1/2 * (A + I) * D^-1/2
    adj = adj + torch.eye(adj.size(0))
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def prepare_facebook_data(file_path, target_attr):
    G = nx.read_gml(file_path)
    G = nx.convert_node_labels_to_integers(G)

    label_attr = "major_index" if target_attr == "major" else target_attr

    valid_nodes = [
        n for n in G.nodes if G.nodes[n].get(label_attr) not in (None, 0, "0")
    ]
    G_sub = G.subgraph(valid_nodes).copy()

    # Features: use all attributes except the target label
    all_attrs = ["gender", "year", "dorm", "major_index"]
    feature_attrs = [a for a in all_attrs if a != label_attr]

    # One-hot encoding for features
    node_data = []
    for n in G_sub.nodes:
        feat_dict = {
            attr: str(G_sub.nodes[n].get(attr, "missing")) for attr in feature_attrs
        }
        node_data.append(feat_dict)

    df_feat = pd.get_dummies(pd.DataFrame(node_data))
    features = torch.FloatTensor(df_feat.values)

    # Label Mapping
    labels_raw = [G_sub.nodes[n][label_attr] for n in G_sub.nodes]
    unique_labels = sorted(list(set(labels_raw)))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = torch.LongTensor([label_map[label] for label in labels_raw])

    adj = nx.to_numpy_array(G_sub)
    return adj, labels, features


def run_recovery(school_file, target_attr, fractions=[0.1, 0.2, 0.3]):
    adj_raw, labels, features = prepare_facebook_data(school_file, target_attr)
    adj_norm = normalize_adj(torch.FloatTensor(adj_raw))

    num_nodes = labels.size(0)
    final_results = {}

    for f in tqdm(fractions, desc=f"{target_attr} missing", leave=False):
        indices = np.random.permutation(num_nodes)
        num_remove = int(num_nodes * f)
        unlabeled_idx = indices[:num_remove]
        labeled_idx = indices[num_remove:]

        model = GCN(
            nfeat=features.shape[1], nhid=16, nclass=int(labels.max()) + 1, dropout=0.5
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # Training
        model.train()
        for _ in tqdm(
            range(200), desc=f"{target_attr} f={f:.1f}", leave=False, unit="epoch"
        ):
            optimizer.zero_grad()
            output = model(features, adj_norm)
            loss = F.nll_loss(output[labeled_idx], labels[labeled_idx])
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        output = model(features, adj_norm)
        preds = output[unlabeled_idx].max(1)[1]

        acc = preds.eq(labels[unlabeled_idx]).sum().item() / num_remove

        # Labels are categorical indices, so MAE is meaningless; compare against
        # always predicting the most common label among the labeled nodes instead.
        majority = labels[labeled_idx].mode().values.item()
        baseline = (labels[unlabeled_idx] == majority).sum().item() / num_remove

        final_results[f] = f"{acc:.3f} (baseline: {baseline:.3f})"

    return final_results


# Execute for Question 5(c) and 5(d)
school = "data/Amherst41.gml"

# school = "data/Harvard1.gml"
# Note: running with Harvard (as an example for a large network) takes significantly longer, and yields the following accuracies:
#   Attribute    0.1    0.2    0.3
# 0      dorm  0.105  0.109  0.112
# 1     major  0.147  0.155  0.161
# 2    gender  0.648  0.612  0.629

attributes = ["dorm", "major", "gender"]
table_data = []

for attr in tqdm(attributes, desc="Attributes", unit="attr"):
    res = run_recovery(school, attr)
    res["Attribute"] = attr
    table_data.append(res)

df = pd.DataFrame(table_data)
print(df[["Attribute", 0.1, 0.2, 0.3]])
