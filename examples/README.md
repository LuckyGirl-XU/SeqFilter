# Examples Built on [DyGLib](https://github.com/yule-BUAA/DyGLib)

[DyGLib](https://github.com/yule-BUAA/DyGLib) is a widely used temporal graph learning library that provides implementations of various state-of-the-art temporal graph learning models. To simplify the process of getting started, we integrate TGB-Seq with DyGLib and demonstrate how to run TGB-Seq datasets using DyGLib.

## Quick Start

For example, to train a DyGFormer model on the GoogleLocal dataset, you can use the appropriate command with the specified arguments for dataset name, model, and training configuration.

```shell
python train_link_prediction.py --dataset_name GoogleLocal --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --gpu 0 --batch_size 200 --dropout 0.1 --sample_neighbor_strategy recent
```

## Modifications to DyGLib

We retain most of DyGLib's features while making the following key modifications for efficiency and ease of use.

### 1. Memory-Based Methods

We optimized the implementation of memory-based methods, including JODIE, DyRep, and TGN, to accelerate the training process. Inspired by the [TGB implementation](https://github.com/shenyangHuang/TGB) of TGN and DyRep, we addressed key inefficiencies observed in DyGLib.

#### Observed Inefficiencies

1. **Memory Updates for All Nodes vs. Selective Nodes**  
   In DyGLib, memory updates are performed for all nodes with raw messages, regardless of whether they are involved in the current batch. In contrast, TGB updates memory only for nodes directly involved in the batch, including the source and destination nodes, and their $k$-hop neighbors (commonly $k = 1$).  
   - **Impact:** DyGLib's approach becomes time-intensive as the number of nodes with raw messages grows during training, particularly for larger datasets. By the end of an epoch, nearly all nodes may require memory updates, leading to significant delays.

2. **Data Structure for Storing Raw Messages**  
   DyGLib stores raw messages in a Python dictionary, where each node’s raw messages are stored as a list. This structure frequently appends or removes elements during training, which is computationally expensive due to Python’s dict and list operations.  
   In contrast, TGB uses a simpler structure, where each node retains only the most recent raw message. This approach aligns with best practices in memory-based methods and significantly reduces the computational overhead.

#### Revisions in Our Implementation

- **Efficient Memory Updates:** Memory updates are performed only for nodes in the current batch (e.g., source and destination nodes and their $k$-hop neighbors).
- **Optimized Raw Message Storage:** Raw messages are stored in tensors instead of Python dictionaries, and only the latest message for each node is retained.

These revisions significantly reduce the training time for memory-based methods, particularly on large datasets. The leaderboard results provided by TGB-Seq team are based on this optimized implementation of DyGLib.

**Attention**: Since we store only the most recent raw message for each node, we treat it directly as part of the model parameters. As a result, there is no need to save or load raw messages when saving or loading the model.

### 2. DyGFormer

We provide an efficient implementation of the `NeighborCooccurrenceEncoder::count_nodes_appearances` function in the DyGFormer model for bipartite datasets. To use this implementation, you can enable the bipartite mode by adding `is_bipartite` to your commands. If you prefer the original implementation, discard `is_bipartite`.

### 3. Minor Modifications

To enhance usability, we added several arguments. For example:

- **`--load_pretrained`:** Load a pre-trained model for further training or evaluation.
- **`--num_epochs 0`:** We allow `num_epochs` to be `0`. This setting will skip the training phase and directly evaluate the model on the test set.
