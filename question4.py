import os
import random
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# 4(b) Link Prediction Metric Implementations


class LinkPrediction(ABC):
    def __init__(self, graph):
        self.graph = graph
        self.N = len(graph)
        self.nodes = list(graph.nodes())

    def neighbors(self, v):
        """Return the set of neighbors for a node."""
        return set(self.graph.neighbors(v))

    @abstractmethod
    def fit(self, u, v):
        """Compute the link predictor metric p for a node pair"""
        raise NotImplementedError("Fit must be implemented")


class CommonNeighbors(LinkPrediction):
    def fit(self, u, v):
        # Score = intersection of neighbor sets
        return len(self.neighbors(u).intersection(self.neighbors(v)))


class Jaccard(LinkPrediction):
    def fit(self, u, v):
        nu = self.neighbors(u)
        nv = self.neighbors(v)
        union_size = len(nu.union(nv))
        if union_size == 0:
            return 0
        # Score = intersection / union
        return len(nu.intersection(nv)) / union_size


class AdamicAdar(LinkPrediction):
    def fit(self, u, v):
        common_nodes = self.neighbors(u).intersection(self.neighbors(v))
        score = 0
        for z in common_nodes:
            deg = self.graph.degree(z)
            if deg > 1:
                # Score = sum of 1/log(degree) of common neighbors
                score += 1 / np.log(deg)
        return score


def evaluate_link_predictor(file_name, fraction=0.1, k_list=[50, 100, 200, 400]):
    # 1. Select graph from Facebook100 dataset
    path = os.path.join("data", file_name)
    G_fb = nx.Graph(nx.read_gml(path))

    # Extract Largest Connected Component to ensure meaningful topology
    lcc_nodes = max(nx.connected_components(G_fb), key=len)
    G_fb = G_fb.subgraph(lcc_nodes).copy()

    # 2. Randomly remove a given fraction f of edges
    all_edges = list(G_fb.edges())
    num_to_remove = int(len(all_edges) * fraction)
    E_removed_list = random.sample(all_edges, num_to_remove)
    E_removed = set(tuple(sorted(e)) for e in E_removed_list)

    # Create training graph G_train
    G_train = G_fb.copy()
    G_train.remove_edges_from(E_removed_list)

    # Initialize metrics classes
    predictors = {
        "Common Neighbors": CommonNeighbors(G_train),
        "Jaccard": Jaccard(G_train),
        "Adamic/Adar": AdamicAdar(G_train),
    }

    # 3. Compute metrics for ALL node pairs |V|x|V|
    nodes = list(G_train.nodes())
    results = []

    # Identify non-edges in G_train as candidates for prediction
    adj_train = nx.to_numpy_array(G_train)
    candidate_mask = (np.triu(np.ones(adj_train.shape), k=1) == 1) & (adj_train == 0)
    candidate_indices = np.argwhere(candidate_mask)

    for name, predictor in predictors.items():
        # Scoring phase
        scores = []
        for idx_pair in candidate_indices:
            u, v = nodes[idx_pair[0]], nodes[idx_pair[1]]
            score = predictor.fit(u, v)
            scores.append((tuple(sorted((u, v))), score))

        # 4. Sort in decreasing order of confidence
        # This creates the ranked list E_predict
        scores.sort(key=lambda x: x[1], reverse=True)

        # 5. Evaluate for each k
        for k in k_list:
            E_predict_top_k = [pair for pair, score in scores[:k]]

            # Intersection: |E_removed intersect E_predict_top_k|
            tp = len([pair for pair in E_predict_top_k if pair in E_removed])

            # top@k predictive rate = Percentage of correct positive samples
            precision_at_k = tp / k
            recall_at_k = tp / num_to_remove if num_to_remove > 0 else 0

            results.append(
                {
                    "Metric": name,
                    "k": k,
                    "TP": tp,
                    "Precision@k": precision_at_k,
                    "Recall@k": recall_at_k,
                }
            )

    return pd.DataFrame(results)


def process_school(school_file):
    """Helper function to process a single school for parallel execution."""
    print(f"Evaluating: {school_file}")
    try:
        school_df = evaluate_link_predictor(school_file, fraction=0.1)
        return school_df
    except Exception as e:
        print(f"Error processing {school_file}: {e}")
        return None


if __name__ == "__main__":
    data_dir = "data"
    report_dir = "report"

    # Ensure report directory exists
    os.makedirs(report_dir, exist_ok=True)

    # 1. Select a set of 15 schools randomly from the Facebook100 dataset
    # all_files = [f for f in os.listdir(data_dir) if f.endswith(".gml")]
    # selected_schools = random.sample(all_files, 15)

    # 1. Select a predefined set of 15 schools from the Facebook100 dataset
    # Note: running the random selection almost always results in an oom error on my end due to the O(n²) scoring step.
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

    print(f"Starting evaluation on {len(selected_schools)} schools...")

    # 2. Perform calculation on each school in parallel to reduce the lengthy runtime of O(n²) scoring step
    # Note: This was tested on a 24-core machine and reduced runtime to approximately 2minutes for the predefined set.
    # Using the random selection without parallelization still results in an OOM error if one of the larger schools is hit,
    # altough I only have 32GB of RAM available.
    all_evaluations = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_school)(school_file) for school_file in selected_schools
    )

    # Filter out None results (from failed schools)
    all_evaluations = [df for df in all_evaluations if df is not None]
    print(f"Successfully evaluated {len(all_evaluations)} schools")

    # Combine all individual school results into a single master DataFrame
    master_df = pd.concat(all_evaluations, ignore_index=True)

    # 3. Aggregate results and plot
    # We calculate the mean Precision@k across the 15 schools for each metric
    plot_data = (
        master_df.groupby(["Metric", "k"])["Precision@k"].mean().unstack(level=0)
    )

    # Creating the visualization for Question 4(d)
    plt.figure(figsize=(10, 6))
    plot_data.plot(kind="line", marker="o", ax=plt.gca())

    plt.title("Mean Precision@k for Link Prediction (Population: 15 Schools)")
    plt.xlabel("k (Number of Top-Ranked Pairs)")
    plt.ylabel("Mean Precision")
    plt.xticks([50, 100, 200, 400])  # Ensuring the required k values are labeled
    plt.legend(title="Metric")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    # Save figure to report directory
    fig_filename = "question4_link_prediction_precision.png"
    fig_path = os.path.join(report_dir, fig_filename)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    plt.show()

    # Output numerical summary
    print("\n--- Final Aggregated Results (Mean Across 15 Schools) ---")
    print(master_df.groupby(["Metric", "k"])[["Precision@k", "Recall@k"]].mean())
