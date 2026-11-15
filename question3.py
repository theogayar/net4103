import os
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

data_dir = "data"
report_dir = "report"
all_files = [f for f in os.listdir(data_dir) if f.endswith(".gml")]

# Ensure report directory exists
os.makedirs(report_dir, exist_ok=True)

# Note: The FB100 dataset doesn't include "status", but rather "student_fac" as an attribute.
target_attributes = ["student_fac", "major_index", "dorm", "gender"]


def process_graph(file_name):
    """
    Loads a single GML file and calculates assortativity coefficients.
    Handles undefined results (zero variance) by returning np.nan.
    """
    # Suppress warnings for undefined assortativity due to zero variance.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            G = nx.read_gml(os.path.join(data_dir, file_name))
            n = G.number_of_nodes()

            # Vertex Degree Assortativity
            r_degree = nx.degree_assortativity_coefficient(G)

            row = {"name": file_name, "n": n, "degree": r_degree}

            # Attribute Assortativity
            for attr in target_attributes:
                try:
                    r_attr = nx.attribute_assortativity_coefficient(G, attr)
                    # Ensure cases with no attribute variance are treated as NaN
                    row[attr] = r_attr if not np.isnan(r_attr) else np.nan
                except (ZeroDivisionError, ValueError, KeyError):
                    row[attr] = np.nan
            print(f"Processed {file_name}")
            return row
        except Exception:
            print(f"Error processing {file_name}")
            return None


# Parallel execution utilizing all available CPU threads
results = Parallel(n_jobs=-1)(delayed(process_graph)(f) for f in all_files)

# Create DataFrame and calculate global population means
df = pd.DataFrame([r for r in results if r is not None])
plot_attributes = ["student_fac", "major_index", "degree", "dorm", "gender"]

print("\n--- Mean Assortativity Values Across FB100 ---")
print(df[plot_attributes].mean(numeric_only=True))

# Plotting Question 3(a)
for i, attr in enumerate(plot_attributes):
    valid_data = df[["n", attr]].dropna()

    # Create a new figure for each attribute: 1 row, 2 columns
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Assortativity Analysis: {attr.replace('_', ' ').title()}", fontsize=16
    )

    # 1. Scatter Plot (Assortativity vs. Network Size)
    ax_scatter.scatter(
        valid_data["n"], valid_data[attr], alpha=0.6, s=30, color="tab:blue"
    )
    ax_scatter.set_xscale("log")  # Log-linear axes as required by the assignment
    ax_scatter.axhline(0, color="red", linestyle="--", label="No Assortativity")
    ax_scatter.set_title("Assortativity vs. Size (n)")
    ax_scatter.set_xlabel("Network Size (n)")
    ax_scatter.set_ylabel("Assortativity coefficient (r)")
    ax_scatter.grid(True, which="both", ls="-", alpha=0.2)

    # 2. Histogram (Distribution of r values)
    ax_hist.hist(valid_data[attr], bins=20, color="skyblue", edgecolor="black")
    ax_hist.axvline(0, color="red", linestyle="--")
    ax_hist.set_title("Distribution of r-values")
    ax_hist.set_xlabel("Assortativity coefficient (r)")
    ax_hist.set_ylabel("Frequency (Number of Schools)")

    plt.tight_layout()

    # Save figure to report directory
    fig_filename = f"question3_assortativity_{attr}.png"
    fig_path = os.path.join(report_dir, fig_filename)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    plt.show()
