from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
from collections import defaultdict
from tgb_seq.LinkPred.dataloader import TGBSeqLoader
class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray, neg_samples: np.ndarray = None, split=None ):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        self.split=split
        if neg_samples is not None:
            self.neg_samples = neg_samples
            self.num_neg_samples = neg_samples.shape[1]
            # check if the number of negative samples matches the number of test samples, where the `split` is 2
            if split is not None:
                assert np.where(split==2)[0].shape[0] == neg_samples.shape[0], 'Number of negative samples does not match the number of test samples!'
            else:
                assert len(src_node_ids) == neg_samples.shape[0], 'Number of negative samples does not match the number of test samples!'
        else:
            self.neg_samples = None
            self.num_neg_samples = 0



def get_link_prediction_data(dataset_name: str, val_ratio: float = 0.15, test_ratio: float = 0.15, dataset_path='./processed_data/', use_edge_feat=False,use_node_feat=False):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # TGB-Seq
    if dataset_name in ["ML-20M", "Taobao", "Yelp","GoogleLocal", "Flickr", "YouTube", "Patent", "WikiLink"]:
        data=TGBSeqLoader(dataset_name, dataset_path)
        src_node_ids,dst_node_ids,node_interact_times,test_ns, split = data.src_node_ids, data.dst_node_ids, data.node_interact_times, data.negative_samples, data.split
        if data.edge_features is not None and use_edge_feat:
            edge_raw_features =data.edge_features
            edge_raw_features=np.concatenate([np.zeros((1,edge_raw_features.shape[1])),edge_raw_features],axis=0) # padding 0 for the first edge
            print("Using edge features from the dataset")
        else:
            edge_raw_features = np.zeros((len(src_node_ids)+1, 1))
            print("Using zero edge features")
        if data.node_features is not None and use_node_feat:
            node_raw_features = data.node_features
            node_raw_features=np.concatenate([np.zeros((1,node_raw_features.shape[1])),node_raw_features],axis=0)
            print("Using node features from the dataset")
        else:
            node_raw_features = np.zeros((len(set(src_node_ids) | set(dst_node_ids))+1, 1))
            print("Using zero node features")
        NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
        assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
        assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
        # padding the features of edges and nodes to the same dimension (172 for all the datasets)
        if node_raw_features.shape[1] < NODE_FEAT_DIM:
            node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
            node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
        if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
            edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
            edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

        assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

        print('dataset_name',dataset_name)
        edge_ids = np.arange(1,len(src_node_ids)+1).astype(np.longlong)
        labels=np.zeros(len(src_node_ids))
        train_mask, val_mask, test_mask = data.train_mask,data.val_mask,data.test_mask
        full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels, split = split)

        train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],node_interact_times=node_interact_times[train_mask],edge_ids=edge_ids[train_mask], labels=labels[train_mask])

        val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask],neg_samples=None, split = split[val_mask])

        test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask],neg_samples=test_ns, split = split[test_mask])

        val_test_edgelist=data.edgelist[data.edgelist['split']!=0]
        val_test_data= Data(src_node_ids=val_test_edgelist['src'].values.astype(np.longlong), dst_node_ids=val_test_edgelist['dst'].values.astype(np.longlong),
                        node_interact_times=val_test_edgelist['time'].values.astype(np.float64), edge_ids=np.arange(1,len(val_test_edgelist)+1).astype(np.longlong), labels=np.zeros(len(val_test_edgelist)),neg_samples=test_ns, split=val_test_edgelist['split'].values)
        
    # DGB
    elif dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci']:
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

        NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
        assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
        assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
        # padding the features of edges and nodes to the same dimension (172 for all the datasets)
        if node_raw_features.shape[1] < NODE_FEAT_DIM:
            node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
            node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
        if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
            edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
            edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

        assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

        # get the timestamp of validate and test set
        val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

        src_node_ids = graph_df.u.values.astype(np.longlong)
        dst_node_ids = graph_df.i.values.astype(np.longlong)
        node_interact_times = graph_df.ts.values.astype(np.float64)
        edge_ids = graph_df.idx.values.astype(np.longlong)
        labels = graph_df.label.values

        full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

        # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
        train_mask = node_interact_times <= val_time
        train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                        node_interact_times=node_interact_times[train_mask],
                        edge_ids=edge_ids[train_mask], labels=labels[train_mask])

        val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
        test_mask = node_interact_times > test_time

        # validation and test data
        val_split = np.full(val_mask.sum(),1)
        val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask],split=val_split)
        
        test_split = np.full(test_mask.sum(),2)
        test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask],neg_samples=test_ns, split=test_split)
        
        # We only use Patent's val_test_data since its source nodes interact with all their neighbors at the same timestamps
        val_test_data = None


    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    
    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, None, None, val_test_data

