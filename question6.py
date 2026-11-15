import networkx as nx
import community.community_louvain as louvain
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from networkx.algorithms.community import (
    greedy_modularity_communities,
    asyn_lpa_communities,
    label_propagation_communities,
)


def validate_community_hypothesis(networks):
    results = {}

    for name, G in networks.items():
        # 1. Detect communities using Louvain (Modularity Optimization)
        partition = louvain.best_partition(G)
        detected_labels = [partition[node] for node in G.nodes()]

        # 2. Extract ground truth metadata
        years = [G.nodes[n].get("year", 0) for n in G.nodes()]
        majors = [G.nodes[n].get("major_index", 0) for n in G.nodes()]
        dorms = [G.nodes[n].get("dorm", 0) for n in G.nodes()]

        # 3. Compute Normalized Mutual Information (NMI)
        nmi_year = normalized_mutual_info_score(years, detected_labels)
        nmi_major = normalized_mutual_info_score(majors, detected_labels)
        nmi_dorm = normalized_mutual_info_score(dorms, detected_labels)

        results[name] = {
            "Year": nmi_year,
            "Major": nmi_major,
            "Dorm": nmi_dorm,
            "Size": G.number_of_nodes(),
        }

        print(f"--- Results for {name} ---")
        print(f"NMI (Year): {nmi_year:.4f}")
        print(f"NMI (Major): {nmi_major:.4f}")
        print(f"NMI (Dorm): {nmi_dorm:.4f}\n")

    return results


# Example Execution
def _detect_partition(G, method):
    if method == "louvain":
        partition = louvain.best_partition(G)
        return {n: partition[n] for n in G.nodes()}

    if method == "greedy":
        communities = list(greedy_modularity_communities(G))
        mapping = {}
        for cid, comm in enumerate(communities):
            for n in comm:
                mapping[n] = cid
        return mapping

    if method == "label_prop":
        communities = list(label_propagation_communities(G))
        mapping = {}
        for cid, comm in enumerate(communities):
            for n in comm:
                mapping[n] = cid
        return mapping

    if method == "asyn_lpa":
        communities = list(asyn_lpa_communities(G))
        mapping = {}
        for cid, comm in enumerate(communities):
            for n in comm:
                mapping[n] = cid
        return mapping

    raise ValueError(f"Unknown method: {method}")


def _extract_ground_truth(G):
    return {
        "Year": [G.nodes[n].get("year", 0) for n in G.nodes()],
        "Major": [G.nodes[n].get("major_index", 0) for n in G.nodes()],
        "Dorm": [G.nodes[n].get("dorm", 0) for n in G.nodes()],
    }


def _compute_nmi(G, method):
    """Compute NMI scores for a single graph/method combo."""
    mapping = _detect_partition(G, method)
    detected_labels = [mapping.get(n, -1) for n in G.nodes()]

    truth = _extract_ground_truth(G)

    nmi_year = normalized_mutual_info_score(truth["Year"], detected_labels)
    nmi_major = normalized_mutual_info_score(truth["Major"], detected_labels)
    nmi_dorm = normalized_mutual_info_score(truth["Dorm"], detected_labels)

    return {
        "NMI_Year": nmi_year,
        "NMI_Major": nmi_major,
        "NMI_Dorm": nmi_dorm,
    }


def _resolve_worker_count(n_jobs):
    if n_jobs in (-1, None):
        return os.cpu_count() or 1
    return max(1, int(n_jobs))


def run_selected_experiments(
    data_dir="data",
    out_dir="report",
    n_jobs=-1,
    methods=None,
    max_nodes_for_greedy=5000,
):
    """Run experiments with visible progress and safer defaults."""
    os.makedirs(out_dir, exist_ok=True)

    selected_files = [
        ("Caltech", os.path.join(data_dir, "Caltech36.gml")),
        ("MIT", os.path.join(data_dir, "MIT8.gml")),
        ("UNC", os.path.join(data_dir, "UNC28.gml")),
        ("Harvard", os.path.join(data_dir, "Harvard1.gml")),
    ]

    if methods is None:
        methods = ["louvain", "label_prop", "asyn_lpa"]

    # Load graphs first
    graphs = {}
    for name, path in selected_files:
        if os.path.exists(path):
            graphs[name] = nx.read_gml(path)

    tasks = []
    skipped_tasks = []
    for name in graphs.keys():
        G = graphs[name]
        for method in methods:
            if method == "greedy" and G.number_of_nodes() > max_nodes_for_greedy:
                skipped_tasks.append((name, method))
                continue
            tasks.append((name, method, G))

    results = {name: {} for name in graphs.keys()}

    worker_count = _resolve_worker_count(n_jobs)
    mp_ctx = None
    if os.name == "posix":
        try:
            mp_ctx = mp.get_context("fork")
        except ValueError:
            mp_ctx = None

    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_ctx) as executor:
        future_to_task = {
            executor.submit(_compute_nmi, G, method): (name, method)
            for name, method, G in tasks
        }

        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Computing NMI",
            unit="task",
        ):
            name, method = future_to_task[future]
            results[name][method] = future.result()

    # Print results
    for name in graphs.keys():
        print(f"\n{name} ({graphs[name].number_of_nodes()} nodes)")
        for method in methods:
            if method in results[name]:
                r = results[name][method]
                print(
                    f" {method:12} | Year: {r['NMI_Year']:.4f} | Major: {r['NMI_Major']:.4f} | Dorm: {r['NMI_Dorm']:.4f}"
                )
            elif (name, method) in skipped_tasks:
                print(
                    f" {method:12} | skipped on {graphs[name].number_of_nodes()} nodes (set max_nodes_for_greedy higher to force it)"
                )

    # Plot all results
    fig, axes = plt.subplots(1, len(graphs), figsize=(16, 5))
    if len(graphs) == 1:
        axes = [axes]

    for idx, name in enumerate(graphs.keys()):
        ax = axes[idx]
        x = np.arange(len(methods))
        width = 0.25

        nmi_years = [results[name].get(m, {}).get("NMI_Year", np.nan) for m in methods]
        nmi_majors = [
            results[name].get(m, {}).get("NMI_Major", np.nan) for m in methods
        ]
        nmi_dorms = [results[name].get(m, {}).get("NMI_Dorm", np.nan) for m in methods]

        ax.bar(x - width, nmi_years, width, label="Year", alpha=0.8)
        ax.bar(x, nmi_majors, width, label="Major", alpha=0.8)
        ax.bar(x + width, nmi_dorms, width, label="Dorm", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_title(f"{name}")
        ax.set_ylabel("NMI Score")
        if idx == 0:
            ax.legend()

    plt.tight_layout()
    out_png = os.path.join(out_dir, "nmi_results.png")
    fig.savefig(out_png, dpi=100)
    plt.show()

    return results


if __name__ == "__main__":
    # Run with default fast methods (louvain, label_prop, asyn_lpa)
    # Greedy is skipped on large graphs by default
    results = run_selected_experiments(
        data_dir="data",
        out_dir="report",
        n_jobs=-1,
        methods=None,
    )
