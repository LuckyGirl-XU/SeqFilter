import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from models.EdgeBank import edge_bank_link_prediction
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler, multi_negative_sampler
from utils.DataLoader import Data
from tgb_seq.LinkPred.evaluator import Evaluator 


def evaluate_model_link_prediction_multi_negs(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module, device: str = 'cpu',
                                   num_neighbors: int = 20, time_gap: int = 2000, num_negs=100):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer','SeqFilter' ]:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    evaluator=Evaluator()

    if evaluate_data.neg_samples is not None:
        num_negs = evaluate_data.num_neg_samples
    neg_samples_idx = 0
    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(
            evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]
            to_test_mask=evaluate_data.split[evaluate_data_indices]==2
            test_neg_sample_idx = np.arange(neg_samples_idx, neg_samples_idx + to_test_mask.sum())
            neg_samples_idx += to_test_mask.sum()
            repeated_batch_src_node_ids = np.repeat(batch_src_node_ids, repeats=num_negs, axis=0)
            repeated_batch_dst_node_ids = np.repeat(batch_dst_node_ids, repeats=num_negs, axis=0)
            repeated_batch_dst_node_ids_reshape = repeated_batch_dst_node_ids.reshape(-1, num_negs)
            original_batch_size = batch_src_node_ids.shape[0]
            if evaluate_data.neg_samples is not None:
                # since tgb-seq neg samples are only provided for test sample, 
                test_neg_dst_node_ids = evaluate_data.neg_samples[test_neg_sample_idx]
                not_test_neg_dst_node_ids = multi_negative_sampler(evaluate_neg_edge_sampler, repeated_batch_dst_node_ids_reshape[~to_test_mask],num_negs)
                batch_neg_dst_node_ids = np.zeros((original_batch_size, num_negs), dtype=np.int32)
                batch_neg_dst_node_ids[to_test_mask] = test_neg_dst_node_ids
                batch_neg_dst_node_ids[~to_test_mask] = not_test_neg_dst_node_ids
            else:
                batch_neg_dst_node_ids=multi_negative_sampler(evaluate_neg_edge_sampler, repeated_batch_dst_node_ids_reshape, num_negs)
            batch_neg_dst_node_ids=batch_neg_dst_node_ids.flatten()
            repeated_batch_node_interact_times = np.repeat(batch_node_interact_times, repeats=num_negs, axis=0)

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_node_interact_times=np.concatenate([batch_node_interact_times, batch_node_interact_times, repeated_batch_node_interact_times], axis=0)
                batch_src_node_embeddings, batch_dst_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          neg_dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=num_neighbors)
                batch_neg_src_node_embeddings=torch.repeat_interleave(batch_src_node_embeddings, repeats=num_negs, dim=0)
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                # batch_src_node_embeddings, batch_dst_node_embeddings = \
                #     model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                #                                                       dst_node_ids=batch_dst_node_ids,
                #                                                       node_interact_times=batch_node_interact_times,
                #                                                       edge_ids=batch_edge_ids,
                #                                                       edges_are_positive=True,
                #                                                       num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                # for i in range(len(repeated_batch_src_node_ids)):
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['SeqFilter']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                # for i in range(len(repeated_batch_src_node_ids)):
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=repeated_batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            if to_test_mask.sum() == 0:
                continue
            positive_probabilities = model[1](
                input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid().cpu().numpy()
            # get negative probabilities, Tensor, shape (batch_size * num_negative_samples_per_node, )
            negative_probabilities = model[1](
                input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid().cpu().numpy()
            if to_test_mask.sum() == 0:
                continue
            positive_probabilities = positive_probabilities[to_test_mask]
            negative_probabilities = negative_probabilities.reshape(-1,num_negs)[to_test_mask]
            evaluate_metrics.extend(evaluator.eval(positive_probabilities,negative_probabilities))
            # print(f'evaluate for the {batch_idx + 1}-th batch')

    return {'mrr': np.mean(evaluate_metrics)}


def evaluate_edge_bank_link_prediction(args: argparse.Namespace, train_data: Data, val_data: Data, test_idx_data_loader: DataLoader,
                                       test_neg_edge_sampler: NegativeEdgeSampler, test_data: Data, num_negs=100):
    """
    evaluate the EdgeBank model for link prediction
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_idx_data_loader: DataLoader, test index data loader
    :param test_neg_edge_sampler: NegativeEdgeSampler, test negative edge sampler
    :param test_data: Data, test data
    :return:
    """
    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    train_val_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids]),
                          dst_node_ids=np.concatenate(
                              [train_data.dst_node_ids, val_data.dst_node_ids]),
                          node_interact_times=np.concatenate(
                              [train_data.node_interact_times, val_data.node_interact_times]),
                          edge_ids=np.concatenate(
                              [train_data.edge_ids, val_data.edge_ids]),
                          labels=np.concatenate([train_data.labels, val_data.labels]))

    test_metric_all_runs = []

    evaluator=Evaluator()

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        loss_func = nn.BCELoss()

        # evaluate EdgeBank
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
        assert test_neg_edge_sampler.seed is not None
        test_neg_edge_sampler.reset_random_state()

        test_losses, test_metrics = [], []
        test_idx_data_loader_tqdm = tqdm(test_idx_data_loader, ncols=120)

        if test_data.neg_samples is not None:
            num_negs = test_data.num_neg_samples

        for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):
            test_data_indices = test_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                test_data.src_node_ids[test_data_indices], test_data.dst_node_ids[test_data_indices], \
                test_data.node_interact_times[test_data_indices]
            repeated_batch_src_node_ids = np.repeat(batch_src_node_ids, repeats=num_negs, axis=0)
            if test_data.neg_samples is not None:
                batch_neg_dst_node_ids = test_data.neg_samples[test_data_indices].flatten()
            else:
                _, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(
                    size=len(repeated_batch_src_node_ids))
                repeated_batch_dst_node_ids = np.repeat(
                    batch_dst_node_ids, repeats=num_negs, axis=0)
                # collision check
                pos_dst_node_ids_reshape = repeated_batch_dst_node_ids.reshape(-1, num_negs)
                batch_neg_dst_node_ids_reshape = batch_neg_dst_node_ids.reshape(
                    -1, num_negs)
                mask = pos_dst_node_ids_reshape == batch_neg_dst_node_ids_reshape
                while np.any(mask):
                    mask_rows = np.where(mask)[0]
                    num_mask_rows = len(mask_rows)
                    _, tmp_negs = test_neg_edge_sampler.sample(
                        size=num_mask_rows*num_negs)
                    batch_neg_dst_node_ids_reshape[mask_rows]=tmp_negs.reshape(-1,num_negs)
                    mask = pos_dst_node_ids_reshape == batch_neg_dst_node_ids_reshape
                batch_neg_dst_node_ids=batch_neg_dst_node_ids_reshape.reshape(-1)
            
            # repeated_batch_node_interact_times = np.repeat(batch_node_interact_times, repeats=num_negs, axis=0)

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (repeated_batch_src_node_ids, batch_neg_dst_node_ids)
            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_val_data.src_node_ids, test_data.src_node_ids[: test_data_indices[0]]]),
                                dst_node_ids=np.concatenate(
                                    [train_val_data.dst_node_ids, test_data.dst_node_ids[: test_data_indices[0]]]),
                                node_interact_times=np.concatenate(
                                    [train_val_data.node_interact_times, test_data.node_interact_times[: test_data_indices[0]]]),
                                edge_ids=np.concatenate(
                                    [train_val_data.edge_ids, test_data.edge_ids[: test_data_indices[0]]]),
                                labels=np.concatenate([train_val_data.labels, test_data.labels[: test_data_indices[0]]]))

            # perform link prediction for EdgeBank
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                                                                                       positive_edges=positive_edges,
                                                                                       negative_edges=negative_edges,
                                                                                       edge_bank_memory_mode=args.edge_bank_memory_mode,
                                                                                       time_window_mode=args.time_window_mode,
                                                                                       time_window_proportion=args.test_ratio)

            test_metrics.extend(evaluator.eval(positive_probabilities, negative_probabilities))

        test_metrics = {'mrr': np.mean(test_metrics)}
        # logger.info(f'test loss: {np.mean(test_losses):.8f}')
        for metric_name in test_metrics.keys():
            logger.info(f'test {metric_name}, {test_metrics[metric_name]:.8f}')

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metrics)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "test metrics": {metric_name: f'{test_metrics[metric_name]:.8f}'for metric_name in test_metrics}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(
            save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save negative sampling results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(
            f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.8f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.8f}')
