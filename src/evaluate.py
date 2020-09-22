"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
from models import *
from dataloader import *
from torch.utils.data import DataLoader
from evaluation import RankingPerformance
import itertools
import time


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir containing weights to load")
parser.add_argument('--fold', default=1, help='which fold to be run')
parser.add_argument('--gpu', action="store_true", default=False, help='if would like to do centroid calculation using torch-gpu or not')
parser.add_argument('--evaluation_fn', default='iterate_sessions', help='function how to iterate over the test data, iterate_sessions or iterate_minibatch')

SEED = 12345
INF = np.inf


def evaluate_inner_loop(model, params, sample_batched, device='cpu'):
    """ Evaluate the model on a sample minibatch, calls the model and returns the model output and target values
    Args:
       model: (torch.nn.Module) the neural network
       params: (Params) hyperparameters/arguments
       sample_batched: (dict) minibatches of data
       device: (string) cpu or cuda device

    Returns:
       output_batch: torch tensor of shape (n_minibatch, 1) - scores, output of (torch.nn.Module) the neural network
       output_probs_batch: torch tensor of shape (n_minibatch, 1) - probabilitie, output of (torch.nn.Module) the neural network
       y: torch tensor of shape (n_minibatch, 1) - relevance labels
    """
    # fetch the next evaluation batch
    x_i, pos_u, neg_u, y = sample_batched['x'].to(params.tensortype).to(device), sample_batched[
        'pos_centroid'].to(params.tensortype).to(device), sample_batched['neg_centroid'].to(params.tensortype).to(device), sample_batched['y'].to(params.tensortype).to(
        device)
    # compute model output
    output_batch, output_probs_batch = model(x_i, None, u_pos=pos_u, u_neg=neg_u, train=False)
    return (output_batch, output_probs_batch), y


def rank_scores(params, session_ids_accu, scores_accu, y_accu):
    """
    Args:
        params: (Params) hyperparameters/arguments
        session_ids_accu: ndarray of shape (n_items, 1)
        scores_accu: ndarray of shape (n_items, 1)
        y_accu: ndarray of shape (n_items, 1)

    Returns:
        perf_eval_scores: (dict) average ranking scores over queries
    """
    strt_time = time.time()
    assert len(session_ids_accu) == len(scores_accu)
    assert len(session_ids_accu) == len(y_accu)
    grouped_data = itertools.groupby(session_ids_accu.flatten())
    results = np.empty((0, params.rank_list_length), dtype=np.float16())
    strt = 0
    for idx, (key, grp) in enumerate(grouped_data):
        len_si = len(list(grp))
        score_session_i = scores_accu[strt:strt + len_si]
        y_session_i = y_accu[strt:strt + len_si]
        reranked_list = utils.rerank_items(score_session_i, y_session_i, rank_list_length=params.rank_list_length)
        results = np.vstack((results, reranked_list.T))
        strt += len_si
    end_time = time.time()
    time_elapsed = end_time - strt_time
    logging.info('time elapsed: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time_elapsed))))
    logging.info('results.shape: {}'.format(results.shape))
    perf_eval = RankingPerformance(results, eps=params.epsilon, data_name=params.data_name)
    perf_eval.compute_all_scores()
    logging.info(perf_eval.scores)
    return perf_eval.scores


def evaluate(model, loss_fn, data_iterator, params, tb_writer=None, device='cpu', epoch=None):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
        tb_writer: (SummaryWriter) tensorboard writer
        device: (string) cpu or cuda device
        epoch: (int) epoch count
    """
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        # summary for current eval loop
        summ = []
        # sessions ids for current eval loop
        session_ids_mtx = np.empty((0, 1))
        # scores for current eval loop
        scores_mtx = np.empty((0, 1))
        # labels for current eval loop
        y_mtx = np.empty((0, 1))
        # compute metrics over the dataset
        for i, sample_batched in enumerate(data_iterator):
            # compute scores and probabilities for the current sample
            (output_batch, output_probs_batch), y = evaluate_inner_loop(model, params, sample_batched, device=device)
            loss = loss_fn(output_batch, y)
            # accumulate for session based evaluation
            session_ids_mtx = np.vstack((session_ids_mtx, sample_batched['session_id']))
            scores_mtx = np.vstack((scores_mtx, output_probs_batch.cpu().numpy()))
            y_mtx = np.vstack((y_mtx, y.cpu().numpy()))
            # compute all metrics on this batch
            summary_batch = {'loss':loss.item()}
            summ.append(summary_batch)
        # compute rank scores 
        rank_metrics_mean = rank_scores(params, session_ids_mtx, scores_mtx, y_mtx)
        # compute mean of all metrics in summary
        metrics_mean1 = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_mean = {**rank_metrics_mean, **metrics_mean1}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Eval metrics : " + metrics_string)
    if tb_writer:
        tb_writer.add_scalar('Val/Loss', metrics_mean['loss'], epoch)
        tb_writer.add_pr_curve('Val/PR', y_mtx, scores_mtx, epoch)  # needs tensorboard 0.4RC or later
        tb_writer.add_scalars('Val/Scores', metrics_mean, epoch)
    return metrics_mean


def evaluate_by_iterating_minibatch(model, params, test_dataloader, device='cpu', return_item_id=False):
    """
    Args:
        model: (torch.nn.Module) the network
        params: (Params) hyperparameters/arguments
        test_dataloader: (generator) a generator that generates minibatches of data and labels
        device: (string) cpu or cuda device
        return_item_id: (bool) if return_item_id is True, then evaluate_by_iterating_minibatch returns each item id

    Returns:
        mean_scores: (dict) average ranking scores over queries
        query_level_scores: (dict) query-level ranking scores
        query_ids_accu: ndarray of shape (n_queries, 1)
    """
    strt_time = time.time()
    session_ids_accu = np.empty((0, 1))
    scores_accu = np.empty((0, 1))
    y_accu = np.empty((0, 1))
    with torch.no_grad():
        for i, sample_batched in enumerate(test_dataloader):
            session_ids_accu = np.vstack((session_ids_accu, sample_batched['session_id']))
            (output_batch, output_probs_batch), y = evaluate_inner_loop(model, params, sample_batched, device=device)
            scores_accu = np.vstack((scores_accu, output_probs_batch.cpu().numpy()))
            y_accu = np.vstack((y_accu, y.cpu().numpy()))
    grouped_data = itertools.groupby(session_ids_accu.flatten())
    results = np.empty((0, params.rank_list_length), dtype=np.float16())
    strt = 0
    res_sesssion_ids = []
    for idx, (key, grp) in enumerate(grouped_data):
        len_si = len(list(grp))
        res_sesssion_ids.append(key)
        score_session_i = scores_accu[strt:strt + len_si]
        y_session_i = y_accu[strt:strt + len_si]
        reranked_list = utils.rerank_items(score_session_i, y_session_i, rank_list_length=params.rank_list_length)
        results = np.vstack((results, reranked_list.T))
        strt += len_si
    end_time = time.time()
    time_elapsed = end_time - strt_time
    logging.info('time elapsed: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time_elapsed))))
    logging.info('results.shape: {}'.format(results.shape))
    if return_item_id:
        assert len(results) == len(res_sesssion_ids)
        return results, res_sesssion_ids
    return results


def initialize_model(params, device='cpu'):
    """
        Parameters
        ----------
        params -- (hyper)parameters of the current model

        Returns
        -------
        model -- initialized model according to the model specified in params
    """
    model = AdaptivePointwisePairwise(params, device=device)
    #
    if torch.cuda.device_count() > 1:
        logging.info("It is using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    #
    return model


def initialize_dataloader(params, fold, return_train_val=False):
    """
    Args:
        params: (hyper)parameters of the data, model
        fold: (int) which data fold
        return_train_val: (boolean)
    Returns:
        test_set
    """
    # getting data in minibatches
    test_batch = RecommendationDataset(params.test_data_file % fold, params.test_centroid_file % fold, params.feature_mtx, return_item_id=True)
    if return_train_val:
        train_batch = RecommendationDataset(params.train_data_file.replace('pw_', '') % fold, params.feature_mtx, rank_list_length=params.rank_list_length, return_item_id=True)
        val_batch = RecommendationDataset(params.val_data_file % fold, params.val_centroid_file % fold, params.feature_mtx, return_item_id=True)
    if return_train_val:
        return test_batch, val_batch, train_batch
    else:
        return test_batch


def evaluate_main(args_model_dir, args_fold=1, args_gpu=True, args_restore_file='best', return_instance_score=False, set_logger=True, return_item_id=False):
    """
    Args:
        args_model_dir: (string) directory containing config
        fold: (int) data fold (e.g. 1)
        args_gpu: (bool) if it's True forces to use gpu device
        args_restore_file: (string) name of file to restore from (without its extension .pth.tar)
        return_instance_score: boolean if it's True function returns the instance scores e.g. NDCG per query
        set_logger: boolean if it's true it setups the logger
        return_item_id: boolean if it's true it returns item ids

    Returns:
    instance_scores - if return_instance_score set True, then it returns instance scores e.g. P@1,5,10, NDCG@1,5,10 per query
    """
    json_path = os.path.join(args_model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    fold = args_fold
    args_model_dir = os.path.join(args_model_dir, 'fold%s/' % fold)
    params.tensortype = torch.float32
    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('using {}'.format(device))
    # Set the random seed for reproducible experiments
    torch.manual_seed(SEED)
    if params.cuda: torch.cuda.manual_seed(SEED)
    #
    if set_logger:
        # reset logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Get the logger
        utils.set_logger(os.path.join(args_model_dir, 'evaluate.log'))
        print('log path: {}'.format(os.path.join(args_model_dir, 'evaluate.log')))
        # Create the input data pipeline
        logging.info("Creating the dataset...")
        print('first line written to log!!')
    # load data
    if args_gpu:
        assert device != 'cpu'
    # Define the model and dataset
    test_batch = initialize_dataloader(params, fold)
    model = initialize_model(params, device=device)
    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args_model_dir, args_restore_file + '.pth.tar'), model)
    model = model.cuda() if params.cuda else model
    model.eval()  # we do not want to accumulate gradient
    #
    logging.info("Starting evaluation for the params: {}".format(params.__dict__))
    test_dataloader = DataLoader(test_batch, batch_size=params.batch_size, shuffle=False, num_workers=params.num_worker)
    if return_item_id:
        results, item_ids = evaluate_by_iterating_minibatch(model, params, test_dataloader, device=device, return_item_id=return_item_id)
    else:
        results = evaluate_by_iterating_minibatch(model, params, test_dataloader, device=device)
    #
    perf_eval = RankingPerformance(results, eps=params.epsilon, compute_instance_scores=return_instance_score, data_name='recsys')
    if return_instance_score:
        if return_item_id:
            perf_eval.instance_scores['session_id'] = item_ids
        return perf_eval.instance_scores
    else:
        f_res = open(args_model_dir + 'rank_results_fold%s.csv' % fold, 'a')
        perf_eval.write_cv_fold_scores(f_res, header=True, fold=fold)
        logging.info(perf_eval.scores)
        test_metrics = perf_eval.scores
        save_path = os.path.join(args_model_dir, "metrics_test_{}.json".format(args_restore_file))
        utils.save_dict_to_json(test_metrics, save_path)
        logging.info("- done.")


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    logging.info('args: {}'.format(args))
    evaluate_main(args.model_dir, args_fold=args.fold, args_gpu=args.gpu, args_restore_file=args.restore_file, args_evaluation_fn=args.evaluation_fn, return_instance_score=False)
