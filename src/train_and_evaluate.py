import utils
import os
from dataloader import *
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import trange
import argparse

from models import *
from evaluate import evaluate
SEED = 12345

import numpy as np
np.random.seed(SEED)
import logging


def parse_args():
    parser = argparse.ArgumentParser('one fold of computation of (train-val) of click prediction/reranking airs recommendation')
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    parser.add_argument('--fold', default=1, help='which fold to be run')
    parser.add_argument('--nr_folds', default=5, help='number of cross validation folds')
    parser.add_argument('--cv', action="store_true", default=False,
                        help='if would like to apply cross-validation from fold 1 to nr_folds')
    parser.add_argument('--gpu', action="store_true", default=False,
                        help='if would like to do centroid calculation using torch-gpu or not')
    return parser.parse_args()


def inner_loop_data_load_model_call(sample_batched, params, model, device='cpu'):
    """
    Loads minibatch data and calls the model and returns the model output and target values
    Parameters
    ----------
    sample_batched: (dict) minibatches of data
    params: (Params) hyperparameters/arguments
    model: (torch.nn.Module) the architecture
    device: (string) cpu or cuda device

    Returns
    -------
    output_batch: torch tensor of shape (n_minibatch, 1) - scores, output of (torch.nn.Module)
    output_probs_batch: torch tensor of shape (n_minibatch, 1) - probabilities, output of (torch.nn.Module)
    y: torch tensor of shape (n_minibatch, 1) - relevance labels [0, 1]
    """
    # Send the data and label to the device
    x_i, x_j, pos_u, neg_u, y = sample_batched['x_i'].to(params.tensortype).to(device), sample_batched['x_j'].to(params.tensortype).to(device), sample_batched['pos_centroid'].to(params.tensortype).to(device), sample_batched['neg_centroid'].to(params.tensortype).to(device), sample_batched['y'].to(params.tensortype).to(device)
    output = model(x_i, x_j, u_pos=pos_u, u_neg=neg_u)
    output_batch, output_probs_batch = output
    return output_batch, output_probs_batch, y


def inner_loop_loss_calculate_and_weight_update(model, model_output, optimizer, loss_fn, params, summ, t, i, running_loss, loss_avg, accu_y, accu_probs, epoch=None):
    """
    Args:
        model: (torch.nn.Module) the network
        model_output: (output_batch, output_probs_batch, y) output of inner_loop_data_load_model_call function
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        params: (Params) hyperparameters/arguments
        summ: (list) metric summary
        t: (trange) of tqdm for progress bar
        i: (int) iteration count
        running_loss: (float) running loss
        loss_avg: (float) average loss value
        accu_y: torch tensor of shape (n_minibatch*epoch, 1)
        accu_probs: torch tensor of shape (n_minibatch*epoch, 1)
        epoch: (int) epoch count

    """
    output_batch, output_probs_batch, y = model_output
    # calculating the loss between prediction and target
    loss = loss_fn(output_batch, y)
    #
    if params.l1_regularization:
        # adding l1 norm of the weights in the loss
        l1_reg = None  # accumulate l1 norms
        for W in model.parameters():
            if l1_reg is None:
                l1_reg = W.norm(1)
            else:
                l1_reg = l1_reg + W.norm(1)
        loss += params.reg_lambda * l1_reg  # adding l1 weight regularization
    # calculating gradients for back propagation
    loss.backward()
    # performs updates using calculated gradients
    optimizer.step()
    # if the model is weighted dot product ->
    if params.positive_weight_constraint:
        for p in model.parameters():
            p.data.clamp_(0)
    running_loss += loss.item()  # accumulating loss
    # print statistics
    accu_y = torch.cat((accu_y, y), 0)
    accu_probs = torch.cat((accu_probs, output_probs_batch.detach()), 0)
    if i % params.save_summary_steps == params.save_summary_steps - 1:
        loss = running_loss / params.save_summary_steps  # average loss of "params.save_summary_steps" number of iterations
        logging.info('epoch: %d, iteration: %5d loss: %.3f' % (epoch + 1, i + 1, loss))
        running_loss = 0.0
        summary_batch = {'loss':loss}
        summ.append(summary_batch)
        #
    # update the average loss
    loss_avg.update(loss)  #
    t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
    return model, summ, t, running_loss, loss_avg, accu_y, accu_probs


def train_biliniear_weight(model, optimizer, loss_fn, dataloader, params, epoch=None, device='cpu'):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        epoch: (int) epoch count
        device: (string) cpu or cuda device
    """
    # set model to training mode
    model.train()
    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    # Use tqdm for progress bar
    num_steps = len(dataloader)
    t = trange(num_steps)
    running_loss = 0.0
    #
    accu_y = torch.tensor([]).to(device)
    accu_probs = torch.tensor([]).to(device)
    for i, sample_batched in zip(t, dataloader):
        # zero the parameter gradients -- clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        # forward
        model_output = inner_loop_data_load_model_call(sample_batched, params, model, device=device)
        # backward + optimize
        model, summ, t, running_loss, loss_avg, accu_y, accu_probs = inner_loop_loss_calculate_and_weight_update(model, model_output, optimizer, loss_fn, params, summ, t, i, running_loss, loss_avg, accu_y, accu_probs, epoch=epoch)
        tb_graph = False  # add only once
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean


def main_train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, params, model_dir, restore_file=None, tb_writer=None, device='cpu'):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'labels'
        val_data: (dict) validation data with keys 'data' and 'labels'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
        tb_writer: (SummaryWriter) tensorboard writer
        device: (string) cpu or cuda device
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        checkpoint_dict = utils.load_checkpoint(restore_path, model, optimizer)
        epoch_start_ind = checkpoint_dict['epoch']
    else:
        epoch_start_ind = 0
    best_val_score = np.inf # 0.0 9f accuracy is used then it's 0.0 and best value is compared >=
    #
    num_steps_train = len(train_data)
    #
    for epoch in range(epoch_start_ind, params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        train_metrics_mean = train_biliniear_weight(model, optimizer, loss_fn, train_data, params, epoch=epoch, device=device)
        # Evaluate for one epoch on validation set
        print('starting to evaluate')
        val_metrics = evaluate(model, loss_fn, val_data, params, device=device, epoch=epoch)
        #
        val_score = val_metrics['loss']
        is_best = val_score <= best_val_score
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir, save_last=True)
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_score = val_score
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
        if tb_writer:
            tb_writer.add_scalar('Train/Loss', train_metrics_mean['loss'], epoch)
            tb_writer.add_scalars('Train/Scores', train_metrics_mean, epoch)
            tb_writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            tb_writer.add_scalars('Val/Scores', val_metrics, epoch)
        #
        print('Epoch: {} | Validation loss: {}'.format(epoch, val_metrics['loss']), flush=True)
        print('Validation loss: {}'.format(val_metrics['loss']), flush=True)
    logging.info('done training and validation.')


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
    model = model.cuda() if params.cuda else model
    #
    if torch.cuda.device_count() > 1:
        logging.info("It is using", torch.cuda.device_count(), "GPUs!")  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    #
    return model


def initialize_dataloader(params, fold, return_item_id=False):
    """
    Args:
        params: (Params) hyperparameters/arguments
        fold: (int) data fold
        return_item_id: (bool)
    Returns:
        train_dataloader: (torch.utils.data.DataLoader) a generator that generates minibatches of train set
        val_dataloader: (torch.utils.data.DataLoader) a generator that generates minibatches of validation set
    """
    # getting train data in minibatches
    train_batch = RecommendationDatasetPairwise(params.train_data_file % fold, params.feature_mtx, train=True, return_item_id=return_item_id)
    train_dataloader = DataLoader(train_batch, batch_size=params.batch_size, shuffle=params.shuffle_data, num_workers=params.num_worker, sampler=None)
    # getting val data in minibatches
    val_batch = RecommendationDataset(params.val_data_file % fold, params.val_centroid_file % fold, params.feature_mtx, return_item_id=True)
    val_dataloader = DataLoader(val_batch, batch_size=params.batch_size, shuffle=False, num_workers=params.num_worker)
    #
    return train_dataloader, val_dataloader


def initialize_loss_and_optimizer(params, model):
    """
    Args:
        params: (Params) hyperparameters/arguments
        model: (torch.nn.Module) the network

    Returns:
        criterion: (nn.Module) that takes batch_output and batch_labels and computes the loss for the batch
        optimizer: (torch.optim) torch optimization alg. function that holds the current state and updates the parameters
    """
    criterion = nn.BCEWithLogitsLoss()
    if params.l2_regularization:
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.reg_lambda)
    else:
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    return criterion, optimizer


def main(args):
    fold = args.fold
    #
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # use GPU if available
    params.cuda = torch.cuda.is_available()
    # Set the random seed for reproducible experiments
    params.tensortype = torch.float32
    torch.manual_seed(SEED)
    if params.cuda: torch.cuda.manual_seed_all(SEED)
    #
    # model directory for this fold of data
    args.model_dir = os.path.join(args.model_dir, 'fold%s/' % fold)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    # reset logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    #
    # set tensorboard logs
    parent_dir = [folder for folder in args.model_dir.split('/') if 'experiment' in folder][0]
    tb_dir = args.model_dir.replace(parent_dir, parent_dir + '/tb_logs').replace('/fold', '_fold')
    logging.info('Saving tensorboard logs to {}'.format(tb_dir))
    tb_writer = SummaryWriter(tb_dir)
    #
    # Create the input data pipeline
    logging.info("Loading the datasets...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('using {}'.format(device))
    if args.gpu:
        assert device != 'cpu'
    #
    train_dataloader, val_dataloader = initialize_dataloader(params, fold)
    model = initialize_model(params, device=device)
    #
    criterion, optimizer = initialize_loss_and_optimizer(params, model)
    #
    logging.info('parameters: {}'.format(params.__dict__))  # log parameters
    #
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    main_train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, criterion, params,
                            args.model_dir,
                            args.restore_file, tb_writer=tb_writer, device=device)
    print('SEED: {}'.format(SEED))
    logging.info("- done.")


if __name__ == '__main__':
    args = parse_args()
    if not args.dont_seed_fold:
        SEED = SEED * int(args.fold)
        np.random.seed(SEED)
        print('SEED after seed*fold', SEED)
    logging.info('args: {}'.format(args))
    main(args)