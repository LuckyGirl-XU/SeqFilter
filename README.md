## SeqFilter: NeurIPS 2025 Submission
The **SeqFilter** is designed to address the challenges of complex sequential dynamics in temporal graph learning, which existng temporal graph neural networks (GNNs) achieves MRR performance lower than 60%.

- SeqFilter is built on [TGB-Seq](https://github.com/TGB-Seq/TGB-Seq.git) for a fair comparison in time efficiency and memory efficiency.
- Following TGB-Seq, SeqFilter adopts a standardized evaluation protocol with fixed dataset splits, generating 100 negative samples per test instance and computing the MRR metric for consistent and reliable performance assessment.

## Datasets
We collect eight temporal graphs with sequential dynamics from the newly released TGB-Seq benchmark (ICLR 2025). These datasets range in size from 2 million to 34 million events. Processed versions are available on [Google Drive](https://drive.google.com/drive/folders/1qoGtASTbYCO-bSWAzSqbSY2YgHr9hUhK?usp=sharing), and the original raw datasets can be accessed [here](https://drive.google.com/drive/folders/1_WkYtmpGtxxf2XzzLlOzyzn6WUFkiGD-?usp=sharing).
- ML-20M
- Taobao
- Yelp
- GoogleLocal
- Flickr
- YouTube
- Patent
- WikiLink


#### Requirements

- python>=3.9
- numpy=1.21.5
- pandas>=2.2.3
- pytorch>=2.5.0
- tqdm=4.64.1
- scipy=1.7.3
- scikit-learn=1.0.2


## Usage

### Quick Start
Get started with SeqFilter using this quick-start example. Just follow the commands below to begin your journey with SeqFilter! 🚀🚀🚀

```shell
python examples/train_link_prediction.py --dataset_name GoogleLocal --model_name SeqFilter --gpu 0 --batch_size 200 --sample_neighbor_strategy recent --energy_threshold 0.3
```

### Dataloader

For example, to load the Flickr dataset to `./data/`, run the following code:
```python
from tgb_seq.LinkPred.dataloader import TGBSeqLoader
data=TGBSeqLoader("Flickr", "./data/")
```
Then, Flickr.csv and Flickr_test_ns.npy will be downloaded from Hugging Face automatically into `./data/Flickr/`. The arrays of source nodes, destination nodes, interaction times, negative destination nodes for the test set can be accessed as follows.

```python
src_node_ids=data.src_node_ids
dst_node_ids=data.dst_node_ids
node_interact_times=data.node_interact_times
test_negative_samples=data.negative_samples
```

### Evaluator
Up to now, all the TGB-Seq datasets are evaluated by the MRR metric. The evaluator takes `positive_probabilities` with size as `(batch_size,)` and `negative_probabilities` with size as `(batch_size x number_of_negatives)` as inputs and outputs the rank of eash positive sample with size as `(batch_size)`.
```python
from tgb_seq.LinkPred.evaluator import Evaluator 
evaluator=Evaluator()
result_dict=evaluator.eval(positive_probabilities,negative_probabilities)
```
