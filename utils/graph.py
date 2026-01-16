# utils/graph.py
"""
Graph Sampling Utilities
"""

import torch
from torch_geometric.nn.pool import fps
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn import knn_graph


def compute_feature_map(y, pos, ratio=0.25, batch=None):
    index_down = fps(pos, ratio=ratio, batch=batch)
    pos_down = pos[index_down]
    y_down = y[index_down]
    
    batch_down = batch[index_down] if batch is not None else None
    y_up = knn_interpolate(x=y_down, pos_x=pos_down, pos_y=pos, 
                           batch_x=batch_down, batch_y=batch)

    fm = torch.abs(y - y_up)
    fm = torch.sum(fm, dim=1)
    
    return fm, index_down


def local_sample(x, pos, sample_nodes=512, k=8, ratio=0.25, batch=None, cosine=False, use_pos=False):
    fm, _ = compute_feature_map(x, pos, ratio, batch)
    
    if batch is not None:
        node_indices = torch.arange(x.size(0), device=x.device)
        sampled_indices_list = []
        
        for b in range(batch.max().item() + 1):
            mask = (batch == b)
            if not mask.any(): continue
                
            fm_batch = fm[mask]
            k_sample = min(sample_nodes, fm_batch.size(0))
            _, topk_indices_in_batch = torch.topk(fm_batch, k=k_sample, largest=True)
            
            global_indices = node_indices[mask][topk_indices_in_batch]
            sampled_indices_list.append(global_indices)
            
        sampled_indices = torch.cat(sampled_indices_list, dim=0)
        sampled_batch = batch[sampled_indices]
    else:
        k_sample = min(sample_nodes, x.size(0))
        _, sampled_indices = torch.topk(fm, k=k_sample, largest=True)
        sampled_batch = None
    
    if use_pos:
        local_edge_index = knn_graph(pos[sampled_indices], k=k, loop=False, batch=sampled_batch, cosine=cosine)
    else:
        local_edge_index = knn_graph(x[sampled_indices], k=k, loop=False, batch=sampled_batch, cosine=cosine)
    
    global_edge_index = torch.stack([
        sampled_indices[local_edge_index[0]],
        sampled_indices[local_edge_index[1]]
    ], dim=0)
    
    return global_edge_index, local_edge_index, sampled_indices, sampled_batch


def global_sample(x, pos, ratio=0.25, k=8, batch=None, cosine=False, use_pos=False):
    sampled_indices = fps(pos, ratio=ratio, batch=batch)
    sampled_batch = batch[sampled_indices] if batch is not None else None
    
    if use_pos:
        local_edge_index = knn_graph(pos[sampled_indices], k=k, loop=False, batch=sampled_batch, cosine=cosine)
    else:
        local_edge_index = knn_graph(x[sampled_indices], k=k, loop=False, batch=sampled_batch, cosine=cosine)
    
    global_edge_index = torch.stack([
        sampled_indices[local_edge_index[0]],
        sampled_indices[local_edge_index[1]]
    ], dim=0)
    
    return global_edge_index, local_edge_index, sampled_indices, sampled_batch