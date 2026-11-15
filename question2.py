import os

import matplotlib.pyplot as plt
import networkx as nx

# Define schools and paths
schools = ["Caltech36", "MIT8", "Johns Hopkins55"]
data_dir = "data"
out_dir = "report"
os.makedirs(out_dir, exist_ok=True)

results = {}

for school in schools:
    # 1. Load the graph
    path = os.path.join(data_dir, f"{school}.gml")
    G = nx.read_gml(path)

    # 2. Extract Largest Connected Component (LCC)
    lcc_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc_nodes).copy()

    n = G_lcc.number_of_nodes()
    m = G_lcc.number_of_edges()

    # 3. Compute Metrics
    density = nx.density(G_lcc)
    global_clustering = nx.transitivity(G_lcc)
    mean_local_clustering = nx.average_clustering(G_lcc)

    results[school] = {
        "n": n,
        "m": m,
        "density": density,
        "global_clustering": global_clustering,
        "mean_local_clustering": mean_local_clustering,
        "degrees": [d for _, d in G_lcc.degree()],
        "local_clust": nx.clustering(G_lcc),
    }

    print(
        f"{school}: n={n}, density={density:.4f}, global_C={global_clustering:.4f}, mean_local_C={mean_local_clustering:.4f}"
    )

# Plot A: Degree Distribution
fig_a, axes_a = plt.subplots(1, len(schools), figsize=(15, 4.5))
if len(schools) == 1:
    axes_a = [axes_a]

for i, school in enumerate(schools):
    ax = axes_a[i]
    ax.hist(
        results[school]["degrees"],
        bins=30,
        density=True,
        color="skyblue",
        edgecolor="black",
    )
    ax.set_title(f"Degree Dist: {school}")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Probability")

fig_a.tight_layout()
out_a = os.path.join(out_dir, "question2a.png")
fig_a.savefig(out_a, dpi=150)
print(f"Saved {out_a}")
plt.close(fig_a)

# Plot B: Degree vs Local Clustering
fig_b, axes_b = plt.subplots(1, len(schools), figsize=(15, 4.5))
if len(schools) == 1:
    axes_b = [axes_b]

for i, school in enumerate(schools):
    ax = axes_b[i]
    d = results[school]["degrees"]
    c = list(results[school]["local_clust"].values())
    ax.scatter(d, c, alpha=0.3, s=10)
    ax.set_title(f"Degree vs Clust: {school}")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Local Clustering")

fig_b.tight_layout()
out_b = os.path.join(out_dir, "question2b.png")
fig_b.savefig(out_b, dpi=150)
print(f"Saved {out_b}")
plt.close(fig_b)
