import torch


def average_precision(hit_array):
    """
    compute average precision for retrieval results
    """

    r = len(hit_array)
    ap = 0
    num_rels = 0
    for i in range(r):
        if hit_array[i]:
            ap += hit_array[:i + 1].sum() / (i + 1)
            num_rels += 1

    if num_rels > 0:
        ap /= num_rels
    return ap


def mean_average_precision(query_features, query_labels, gallery_features=None, gallery_labels=None, rank=10, dist="l2"):

    dist_map = {"l2": l2, "cosine": cosine}

    set_diag_inf = False
    if gallery_features is None:
        gallery_features = query_features
        gallery_labels = query_labels
        set_diag_inf = True

    num_features = len(query_labels)
    dist_function = dist_map.get(dist, l2)
    dist_matrix = dist_function(query_features, gallery_features)

    if set_diag_inf:
        dist_matrix.fill_diagonal_(float('inf'))

    idx = dist_matrix.topk(k=rank, dim=-1, largest=False)[1]

    # 일치 여부 판단
    hit_matrix = (gallery_labels[idx[:, :rank]] == query_labels.unsqueeze(dim=-1))

    # average precision 계산
    mean_ap = sum(average_precision(hit_array) for hit_array in hit_matrix) / num_features

    return mean_ap.item()


def precision_at_k(query_features, query_labels, gallery_features=None, gallery_labels=None, rank_list=[1, 5, 10], dist="l2"):

    dist_map = {"l2": l2, "cosine": cosine}

    set_diag_inf = False
    if gallery_features is None:
        gallery_features = query_features
        gallery_labels = query_labels
        set_diag_inf = True

    num_features = len(query_labels)
    dist_function = dist_map.get(dist, l2)
    dist_matrix = dist_function(query_features, gallery_features)

    if set_diag_inf:
        dist_matrix.fill_diagonal_(float('inf'))

    idx = dist_matrix.topk(k=rank_list[-1], dim=-1, largest=False)[1]
    ap_list = []
    for r in rank_list:
        correct = (gallery_labels[idx[:, :r]] == query_labels.unsqueeze(dim=-1)).double().mean(dim=1)
        ap_list.append((torch.sum(correct) / num_features).item())

    return ap_list


def l2(x, y):
    return torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)


def cosine(x, y):
    return (-x @ y.t() + 1)  # (0,2)
