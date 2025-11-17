import json
import torch


def knn_json_to_edge_index(path: str, make_undirected: bool = True) -> torch.Tensor:
    """
    Load knn_train.json and return edge_index [2, E] suitable for GNNs.

    Edges are constructed from i -> j for every neighbor j in indices[i].
    If make_undirected is True, both (i, j) and (j, i) are added.
    """
    with open(path, "r") as f:
        data = json.load(f)

    idxs = data["indices"]  # list of N lists of length k
    N = len(idxs)

    src = []
    dst = []
    for i in range(N):
        for j in idxs[i]:
            src.append(i)
            dst.append(j)
            if make_undirected:
                src.append(j)
                dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index
