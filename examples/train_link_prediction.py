import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.SeqFilter import SeqFilter
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from evaluate_models_utils_mrr import evaluate_model_link_prediction_multi_negs
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, val_test_data = \
        get_link_prediction_data(
            dataset_name=args.dataset_name, dataset_path=args.dataset_path, use_edge_feat=args.use_edge_feat,use_node_feat=args.use_node_feat)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    # new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    # new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)
    if args.model_name in ['DyRep', 'TGN', 'JODIE']:
        test_data = val_test_data
        val_data = val_test_data
    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    # new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    multi_negs_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(
        len(test_data.src_node_ids))), batch_size=args.multi_negs_batch_size, shuffle=False)
    # new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    # val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []
    val_metric_all_runs, test_metric_all_runs = [], []

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    for run in range(args.num_runs):

        set_random_seed(seed=args.seed+run)
        postfix=''
        if args.use_edge_feat:
            postfix+='_e'
        if args.use_node_feat:
            postfix+='_n'
        args.save_model_name = f'{args.model_name}_seed{args.seed+run}_batchsize{args.batch_size}_num_neighbors{args.num_neighbors}_dropout{args.dropout}_sample_neighbor_strategy{args.sample_neighbor_strategy}_numlayers{args.num_layers}{postfix}'
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        os.makedirs(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        fh.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(
                    train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'SeqFilter':
            dynamic_backbone = SeqFilter(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, tau = args.energy_threshold, rhythm_conv =args.rhythm_Conv, structure_conv =args.structure_Conv, device=args.device)
            
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device, bipartite=args.is_bipartite)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer,
                                     learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        save_model_folder = f"{args.save_model_path}/{args.dataset_name}/{args.model_name}/{args.save_model_name}/"
        if not os.path.exists(save_model_folder):
            os.makedirs(save_model_folder)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)
        if args.load_pretrained:
            early_stopping.load_checkpoint(model, map_location='cpu')
        model = convert_to_gpu(model, device=args.device)

        loss_func = nn.BCELoss()
        for epoch in range(args.num_epochs):

            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'SeqFilter']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]
                # t_sample_start = time.perf_counter()
                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(
                    size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids
                # t_sample_end = time.perf_counter()
                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # note that negative nodes do not change the memories while the positive nodes change the memories,
                    # we need to first compute the embeddings of negative nodes for memory-based models
                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_node_interact_times=np.concatenate([batch_node_interact_times,batch_node_interact_times,batch_node_interact_times],axis=0)
                    batch_src_node_embeddings, batch_dst_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          neg_dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          num_neighbors=args.num_neighbors)
                    batch_neg_src_node_embeddings = batch_src_node_embeddings
                elif args.model_name in ['GraphMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)
                elif args.model_name in ['SeqFilter']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)
                elif args.model_name in ['DyGFormer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                # get positive and negative probabilities, shape (batch_size, )
                positive_probabilities = model[1](
                    input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](
                    input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                predicts = torch.cat(
                    [positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(
                    positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)
                train_losses.append(loss.item())
                train_metrics.append(get_link_prediction_metrics(
                    predicts=predicts, labels=labels))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model[0].memory_bank.detach_memory_bank()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank(
                )
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     device=args.device,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap,mode='val')

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # ! This is different from the original DyGLib. We reload the training memory bank in order to store the training memory bank. When testing, we must deal with the edges in val set first.
                model[0].memory_bank.reload_memory_bank(
                    train_backup_memory_bank) 
            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.8f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.8f}')
            logger.info(f'validate loss: {np.mean(val_losses):.8f}')
            for metric_name in val_metrics.keys():
                logger.info(f'validate {metric_name}, {val_metrics[metric_name]:.8f}')
            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics.keys():
                val_metric_indicator.append(
                    (metric_name, val_metrics[metric_name], True))
            early_stop = early_stopping.step(val_metric_indicator, model)
            if early_stop:
                break

        # load the best model
        if early_stopping.load_checkpoint(model, map_location='cpu') < 0:
            sys.exit()
        model = convert_to_gpu(model, device=args.device)
        logger.info(f'get final performance on dataset {args.dataset_name}...')
        test_metrics={}
        # For memory based models, we need to deal with their val set first in the evaluate_model_link_prediction function.
        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   device=args.device,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap,mode='test')
        # reload the model, so that the memory bank is reloaded
        early_stopping.load_checkpoint(model, map_location='cpu')
        model = convert_to_gpu(model, device=args.device)
        # logger.info(f'test loss: {np.mean(test_losses):.8f}')
        for metric_name in test_metrics.keys():
            logger.info(f'test {metric_name}, {test_metrics[metric_name]:.8f}')
        test_metrics_multi_negs = evaluate_model_link_prediction_multi_negs(model_name=args.model_name,
                                                                            model=model,
                                                                            neighbor_sampler=full_neighbor_sampler,
                                                                            evaluate_idx_data_loader=multi_negs_test_idx_data_loader,
                                                                            evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                            evaluate_data=test_data,
                                                                            loss_func=loss_func,
                                                                            device=args.device,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)
        test_metrics.update(test_metrics_multi_negs)
        for metric_name in test_metrics.keys():
            logger.info(f'test {metric_name}, {test_metrics[metric_name]:.8f}')

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metrics)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        result_json = {
            "test metrics": {metric_name: f'{test_metrics[metric_name]:.8f}' for metric_name in test_metrics},
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.dataset_name}/{args.model_name}/"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(
            save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.8f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.8f}')

    sys.exit()
