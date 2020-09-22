import numpy as np
import torch
import torch.nn as nn

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed=seed)


class AdaptivePointwisePairwise(nn.Module):
    """
    Adaptive Pointwise-Pairwise Content-based Recommendation Model
    """
    def __init__(self, params, weight_init=None, eps=1.1920929e-07, device='cpu'):
        super(AdaptivePointwisePairwise, self).__init__()
        self.device = device
        if weight_init:
            self.weight = nn.Parameter(torch.diag(torch.Tensor(weight_init)))
        else:
            self.weight = nn.Parameter(torch.ones(params.item_dim)) # typically weight_dim << item_dim
        if params.beta_init:
            if params.beta_init_one:
                self.beta = nn.Parameter(torch.Tensor(torch.ones(params.item_dim)))
            else:
                self.beta = nn.Parameter(torch.Tensor(torch.ones(params.item_dim)*params.beta_init))
        else:
            self.beta = nn.Parameter(torch.Tensor(torch.rand(params.item_dim)))
            nn.init.normal_(self.beta, 0.0, 0.02)
        self.dropout = False
        self.batch_norm = False
        if params.batch_norm:
            self.batch_norm = True
            self.bn = nn.BatchNorm1d(params.item_dim)
        self.eps = torch.tensor(eps, dtype=torch.float, device=device)
        self.w_gamma = nn.Parameter(torch.ones(params.item_dim))
        self.w_pw = nn.Parameter(torch.ones(params.item_dim))
    #
    def cal_gamma(self, xvd_i, xvd_j):
        """
        calculate gamma
        """
        mul1 = torch.mul(xvd_i, self.w_pw)
        mul = torch.mul(mul1, xvd_j)
        mul_weighted_sum = torch.mv(mul, self.w_gamma)
        gamma = torch.softmax(mul_weighted_sum, dim=0)
        assert gamma.dim() == 1
        return gamma
    #
    def forward(self, x_i, x_j, u_pos, u_neg, train=True):
        """
        Args:
        x_i: torch tensor of shape (batch_size, item_dim) - first item mtx
        x_j: torch tensor of shape (batch_size, item_dim) - second item mtx
        u_pos: torch tensor of shape (batch_size, item_dim) - user positive items mean
        u_neg: torch tensor of shape (batch_size, item_dim) - user negative items mean
        Returns:
            out: torch tensor of shape (batch_size, 1) - scores: last layer output
            probs: torch tensor of shape (batch_size, 1) - probabilities: softmax over last layer output
        """
        beta_neg = torch.mul(u_neg, self.beta)
        pos_neg_cent_difference = u_pos - beta_neg   # taking difference between the negative context and positive (clicked) context
        xv_i = torch.mul(x_i, self.weight)  # dot product of weight matrix and article representation e.g. embedding article in lower dimension
        xvd_i = torch.mul(xv_i, pos_neg_cent_difference)
        if train:
            xv_j = torch.mul(x_j, self.weight)
            xvd_j = torch.mul(xv_j, pos_neg_cent_difference)
        if self.batch_norm:  # apply batchnorm
            xvd_i = self.bn(xvd_i)
            if train:
                xvd_j = self.bn(xvd_j)
        if train:
            gamma = self.cal_gamma(xv_i, xv_j)
            xvd_i_ = torch.sum(xvd_i, dim=1)  # similarity between article and (negative-positive) centroid via dot product of their embedding in lower dimension
            xvd_j_ = torch.sum(xvd_j, dim=1)  # similarity between article and (negative-positive) centroid via dot product of their embedding in lower dimension
            xvd = xvd_i_ - (gamma * xvd_j_)
        else:
            xvd = torch.sum(xvd_i, dim=1)  # similarity between article and (negative-positive) centroid via dot product of their embedding in lower dimension
        probs = torch.sigmoid(xvd)  # probs
        return xvd.unsqueeze(1), probs.unsqueeze(1)

