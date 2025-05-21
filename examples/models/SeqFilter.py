import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft, ifft
import torch.nn.functional as F
from models.modules import TimeEncoder
from utils.utils import NeighborSampler
import math
import random 

import torch
import torch.nn as nn
import numpy as np



class SeqFilter(nn.Module):
    def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler,
                 time_feat_dim, num_layers=1,
                 dropout=0.1, tau = 0.3, rhythm_conv = 4, structure_conv = 8, device='cpu'):
        super(SeqFilter, self).__init__()
        
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.use_feat = 0
        self.tau = tau
        self.rhythm_conv = rhythm_conv
        self.structure_conv = structure_conv


        self.num_channels = self.edge_feat_dim
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        
        self.num_nodes = self.node_raw_features.shape[0]
        self.memory_dim = self.node_feat_dim 
        self.message_dim = self.time_feat_dim
        
        self.memory_bank = MemoryBank(num_nodes=self.num_nodes, memory_dim=self.memory_dim)
        self.memory_updater = RhythmMemoryUpdater(self.memory_bank, message_dim = self.message_dim, memory_dim=self.memory_dim, period = self.rhythm_conv)


        if self.use_feat:
            self.fsts = FSTS(c_in=self.time_feat_dim+self.edge_feat_dim, c_out=self.node_feat_dim, period = self.structure_conv, energy_threshold = self.tau)
        else:
            self.fsts = FSTS(c_in=self.time_feat_dim, c_out=self.node_feat_dim, period = self.structure_conv, energy_threshold = self.tau)

        self.layernorm = nn.LayerNorm(self.memory_dim)
        
    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20, time_gap: int = 2000):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :return:
        """
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.compute_node_temporal_embeddings(node_ids=src_node_ids, node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors, time_gap=time_gap)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.compute_node_temporal_embeddings(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors, time_gap=time_gap)

        return src_node_embeddings, dst_node_embeddings
    
    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         num_neighbors: int = 20, time_gap: int = 2000):

        node_ids_torch = torch.from_numpy(node_ids).to(self.device)
        if not node_ids_torch.dtype == torch.long:
            node_ids_torch = node_ids_torch.long()

        raw_node_features = self.node_raw_features[node_ids_torch]

        time_features = self.time_encoder(torch.from_numpy(node_interact_times).float().unsqueeze(1).to(self.device))
        
        messages = time_features.squeeze(1)

        self.memory_updater.update_memory(node_ids_torch, messages)
        self.memory_bank.set_last_update(node_ids_torch, torch.from_numpy(node_interact_times).float().to(self.device))

        new_mem = self.memory_bank.get_memories(node_ids_torch)

        root_node_time = self.layernorm(new_mem)

        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        nodes_neighbor_time_features = self.time_encoder(
            torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device))
        

        nodes_neighbor_time_features[torch.from_numpy(neighbor_node_ids == 0)] = 0.0

        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
        if self.use_feat:
            neighbor_info = torch.cat([nodes_edge_raw_features, nodes_neighbor_time_features], dim =-1)
        else:
            neighbor_info = nodes_neighbor_time_features

        
        out_t = self.fsts(neighbor_info)  
        
 
        final_emb = out_t + root_node_time
    
        return final_emb

    
    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()




class Global_Local_Filtering(nn.Module):
    def __init__(self, input_dim, output_dim, threshold_param=0.3):
        super(Global_Local_Filtering, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity_threshold = 0.01
        self.threshold_param = threshold_param
        self.scale = 0.02

        self.r1 = nn.Parameter(self.scale * torch.randn(input_dim, output_dim))
        self.i1 = nn.Parameter(self.scale * torch.randn(input_dim, output_dim))
        self.rb1 = nn.Parameter(self.scale * torch.randn(output_dim))
        self.ib1 = nn.Parameter(self.scale * torch.randn(output_dim))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, N, C = x_fft.shape

        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        mean = torch.mean(energy)
        normalized_energy = energy / (mean + 1e-6)
        adaptive_mask = (normalized_energy >= self.threshold_param).float()

        return adaptive_mask 

    def forward(self, neighbor_info):

        neighbor_fft = fft(neighbor_info, dim=1, norm='ortho')



        adaptive_mask = self.create_adaptive_high_freq_mask(neighbor_fft)
        adaptive_mask = adaptive_mask.unsqueeze(-1).expand_as(neighbor_fft)


        low_freq_fft = neighbor_fft * adaptive_mask


        o_real_fft = torch.einsum('bnc,cd->bnd', low_freq_fft.real, self.r1) - \
                     torch.einsum('bnc,cd->bnd', low_freq_fft.imag, self.i1) + self.rb1

        o_imag_fft = torch.einsum('bnc,cd->bnd', low_freq_fft.imag, self.r1) + \
                     torch.einsum('bnc,cd->bnd', low_freq_fft.real, self.i1) + self.ib1

        y_fft = torch.stack([F.relu(o_real_fft), F.relu(o_imag_fft)], dim=-1)
        y_fft = F.softshrink(y_fft, lambd=self.sparsity_threshold)
        y_fft = torch.view_as_complex(y_fft)

        y = ifft(y_fft, dim=1, norm='ortho')
        
        return y.real 
    




class FSTS(nn.Module):
    def __init__(self, c_in, c_out, period, energy_threshold):
        super(FSTS, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.tau = energy_threshold
        
        self.GL_Filter = Global_Local_Filtering(input_dim=c_in, output_dim=c_out, threshold_param = self.tau) 
        self.Local_Correlation = nn.Conv1d(
            in_channels=c_out,
            out_channels=c_out,
            kernel_size=period,
            stride=1,
            padding=period // 2
        )

        
       
    def forward(self, neighbor_info):

        neighbor_info = self.GL_Filter(neighbor_info)


        neighbor_info = neighbor_info.permute(0, 2, 1)


        neighbor_info = self.Local_Correlation(neighbor_info)

        neighbor_out = neighbor_info.permute(0, 2, 1)
        
        neighbor_out = neighbor_out.mean(dim=1)

        return neighbor_out  


class Learnable_Rhythm(nn.Module):
    def __init__(self, input_dim, output_dim, period):
        super(Learnable_Rhythm, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.period = period

        self.learnable_rhythm = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=period,
            padding=period // 2,
            groups=input_dim,  # depthwise
            bias=False
        )

        # Linear projection to output_dim
        self.output_layer = nn.Linear(input_dim, output_dim)



    def forward(self, rhythm_messages):
      

        rhythm_messages = rhythm_messages.permute(0, 2, 1)  
        rhythm_messages = self.learnable_rhythm(rhythm_messages)  
        rhythm_messages = rhythm_messages.permute(0, 2, 1)  

        
        rhythm_messages = self.output_layer(rhythm_messages)  

        return rhythm_messages

class MemoryBank(nn.Module):

    def __init__(self, num_nodes: int, memory_dim: int):
        super(MemoryBank, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        self.node_memories = nn.Parameter(torch.zeros((self.num_nodes, self.memory_dim)), requires_grad=False)
        self.node_last_updated_times = nn.Parameter(torch.zeros(self.num_nodes), requires_grad=False)
        self.layer_norm = nn.LayerNorm(self.memory_dim)
        
        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        self.node_memories.data.zero_()
        self.node_last_updated_times.data.zero_()

    def get_memories(self, node_ids):
        if isinstance(node_ids, np.ndarray):
            node_ids = torch.from_numpy(node_ids).to(self.node_memories.device)
        return self.node_memories[node_ids]

    def set_memories(self, node_ids, updated_node_memories):
        with torch.no_grad():
            self.node_memories.data[node_ids] = self.layer_norm(updated_node_memories)

    def set_last_update(self, node_ids, new_times):
        with torch.no_grad():
            self.node_last_updated_times[node_ids] = new_times

    def get_last_update(self, node_ids):
        return self.node_last_updated_times[node_ids]

class RhythmMemoryUpdater(nn.Module):
    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int, period: int):
        super(RhythmMemoryUpdater, self).__init__()
        self.memory_bank = memory_bank
        self.period = period
        self.message_dim = message_dim
        self.memory_dim = memory_dim
        
        self.rhythm_updater = Learnable_Rhythm(message_dim+memory_dim, memory_dim, self.period)

    def update_memory(self, node_ids, messages):
       
        
        old_mem = self.memory_bank.get_memories(node_ids)
        
        messages = torch.cat([messages, old_mem], dim =-1)

        
        new_mem = self.rhythm_updater(messages.unsqueeze(1)).squeeze(1)
        if new_mem.shape[1] != self.memory_dim:
            new_mem = new_mem.mean(dim=1) 
  
        self.memory_bank.set_memories(node_ids, new_mem)


    

