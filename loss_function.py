import torch
import torch.nn as nn
from chamfer_distance import ChamferDistance


# points and points_reconstructed are n_points x 3 matrices
def compute_chamfer_loss(gt_points, reconstruct_points):
    chamfer_dist = ChamferDistance()
    dist1, dist2 = chamfer_dist(gt_points, reconstruct_points)
    loss = torch.mean(dist1, dim=-1) + torch.mean(dist2, dim=-1)
    return loss
