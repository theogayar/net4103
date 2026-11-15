# NET4103/7431 — Network Science and Graph Learning on Facebook100

Analysis of the Facebook100 dataset (Traud, Mucha & Porter, *Social structure of Facebook networks*, Physica A, 2012): 100 university friendship networks from September 2005, with node attributes (student/faculty status, gender, major, dorm, year).

The full write-up is in [`report/net4103_gaillard.tex`](report/net4103_gaillard.tex).

## Contents

| Script | Topic | Output |
| --- | --- | --- |
| `question2.py` | Degree distributions, density, clustering (Caltech, MIT, Johns Hopkins) | `report/question2a.png`, `report/question2b.png` |
| `question3.py` | Attribute assortativity across all 100 networks | `report/question3_assortativity_*.png` |
| `question4.py` | Link prediction with Common Neighbors, Jaccard, Adamic/Adar (Precision@k, Recall@k) | `report/question4_link_prediction_precision.png` |
| `question4e.py` | Link prediction with a GCN graph autoencoder using one-hot node attributes | printed Precision@k |
| `question5.py` | Missing-label recovery (dorm, major, gender) with a two-layer GCN, compared to a majority-class baseline | printed accuracy table |
| `question6.py` | Community detection (Louvain, label propagation) vs. year/major/dorm, measured with NMI | `report/nmi_results.png` |

Link prediction (Q4, Q4e) runs on a fixed set of 15 small-to-mid-sized schools for reproducibility and to keep memory use manageable.

## Setup

Use **Python 3.13**. On newer versions, `torch_geometric` (used by `question4e.py`) fails at import with `TypeError: _eval_type() missing 1 required positional argument: 'type_params'`.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is used by `question4e.py` when available but is not required.

## Data

The scripts expect the Facebook100 `.gml` files in `data/` (e.g. `data/Caltech36.gml`, `data/MIT8.gml`), about 2.2 GB in total.

## Running

Each script is standalone and is run from the repository root:

```bash
python question2.py
python question3.py
python question4.py
python question4e.py
python question5.py
python question6.py
```

Or run them all in order with `run_all.py`. Plot windows are disabled (figures are still saved to `report/`), and a timing summary is printed at the end:

```bash
python run_all.py                          # everything; stops at the first failure
python run_all.py --only question2 question6
python run_all.py --skip question4e --keep-going
```

## Known limitations

- Scoring every non-adjacent node pair is O(n²) in memory, so `question4.py` runs out of memory on the largest schools. Sparse or sampled candidate scoring would remove this limit.
- Random splits are not seeded, so results vary slightly between runs.
- Missing attribute values (encoded as `0` in the dataset) are kept as their own category when computing NMI in `question6.py`.
