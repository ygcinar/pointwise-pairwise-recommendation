import numpy as np
import logging
from collections import OrderedDict


class RankingPerformance(object):
    def __init__(self, results, eps=2.220446049250313e-16, pre_compute=True, compute_instance_scores=False, data_name=''):
        self.top = np.argmax(results, axis=1)
        self.rank_ideal = -np.sort(-results, axis=1)
        self.results = results
        self.scores = OrderedDict()
        self.cv_scores = OrderedDict()
        self.eps = eps
        self.instance_scores = OrderedDict()
        self.compute_instance_scores = compute_instance_scores
        if pre_compute:
            if data_name == 'outbrain':
                self.compute_outbrain_scores()
            else:
                self.compute_all_scores()
    #
    def compute_mRR(self, k=None):
        if k:
            mRR = np.sum(1. / (self.top[self.top < k]+1)) / len(self.top) #0-based index to ranking starting from 1
            self.scores['mRR' + str(k)] = mRR
            if self.compute_instance_scores:
                in_the_topk = self.top < k
                self.instance_scores['mRR' + str(k)] = []
                for idx, bool_topk in enumerate(in_the_topk):
                    if bool_topk:
                        self.instance_scores['mRR' + str(k)].append(1. / (self.top[idx] + 1))
                    else:
                        self.instance_scores['mRR' + str(k)].append(0)
                assert np.isclose(np.sum(self.instance_scores['mRR'+ str(k)]) / len(self.top), self.scores['mRR' + str(k)])
        else:
            mRR = np.mean(1. / (self.top+1))
            self.scores['mRR'] = mRR
            if self.compute_instance_scores:
                self.instance_scores['mRR'] = 1. / (self.top + 1)  # 0-based index to ranking starting from 1
                assert np.sum(self.instance_scores['mRR']) / len(self.top) == self.scores['mRR']
        return mRR
    #
    def compute_HR_k(self, k):
        HR_k = np.mean(self.top < k)
        self.scores['HR'+str(k)] = HR_k
        if self.compute_instance_scores:
            self.instance_scores['HR'+str(k)] = [float(b) for b in self.top < k]
        return HR_k
    #
    def compute_nDCG_k(self, k):
        len_rank = min(self.results.shape[1], k)
        w = 1. / np.log2(np.arange(2, len_rank+2))
        nDCG_k = np.mean(np.dot(self.results[:, :len_rank], w[:len_rank]) / (self.eps + np.dot(self.rank_ideal[:, :len_rank], w[:len_rank])))
        self.scores['nDCG'+str(k)] = nDCG_k
        if self.compute_instance_scores:
            self.instance_scores['nDCG'+str(k)] = np.dot(self.results[:, :len_rank], w[:len_rank]) / (self.eps + np.dot(self.rank_ideal[:, :len_rank], w[:len_rank]))
        return nDCG_k
    #
    def compute_mAP_k(self, k):
        len_rank = min(self.results.shape[1], k)
        p_acc = 0.
        for idx in range(len_rank):
            p_idx = np.multiply(np.sum(self.results[:, :idx+1], axis=1), self.results[:, idx].T) / (idx+1)
            p_acc = p_acc + p_idx
        ap =  p_acc/len_rank
        if self.compute_instance_scores:
            self.instance_scores['mAP'+str(k)] = ap
        self.scores['mAP' + str(k)] = np.mean(ap)
    #
    def compute_all_scores(self):
        _ = self.compute_mRR()
        _ = self.compute_HR_k(1)
        _ = self.compute_HR_k(4)
        _ = self.compute_HR_k(8)
        _ = self.compute_nDCG_k(8)
        _ = self.compute_nDCG_k(28)
        _ = self.compute_mRR(8)
        _ = self.compute_mRR(28)
        for score_name, score in self.scores.items():
            logging.debug("{}: {}".format(score_name, score))
    #
    def compute_outbrain_scores(self):
        _ = self.compute_mRR()
        _ = self.compute_HR_k(1)
        _ = self.compute_HR_k(3)
        _ = self.compute_HR_k(6)
        _ = self.compute_nDCG_k(3)
        _ = self.compute_nDCG_k(5)
        _ = self.compute_nDCG_k(6)
        _ = self.compute_nDCG_k(12)
        _ = self.compute_mRR(6)
        _ = self.compute_mRR(12)
        _ = self.compute_mAP_k(6)
        _ = self.compute_mAP_k(12)
        for score_name, score in self.scores.items():
            logging.debug("{}: {}".format(score_name, score))
    #
    def compute_all_scores_cv(self, list_scores):
        score_names = list(list_scores[0].keys())
        for score_name in score_names:
            score_list = []
            self.cv_scores[score_name] = {}
            for score_obj in list_scores:
                assert score_names == list(score_obj.keys())
                score_list.append(score_obj[score_name])
            self.cv_scores[score_name]['mean'] = np.mean(score_list)
            self.cv_scores[score_name]['std'] = np.std(score_list)
            self.cv_scores[score_name]['scores'] = score_list
            logging.debug('CV {}: {} ({})'.format(score_name, self.cv_scores[score_name]['mean'], self.cv_scores[score_name]['std']))
    #
    def write_cv_fold_scores(self, o_file, header=False, model_name=None, fold=None):
        if len(self.scores) == 0:
            self.compute_all_scores()
        scores_dict = self.scores
        if header:
            if model_name:
                o_file.write('model_name,')
            o_file.write('fold,' + ','.join(self.scores.keys()) + '\n')
        #
        line = str(fold)
        for score_name in scores_dict.keys():
            line = line + ',' + str(scores_dict[score_name])
        o_file.write(line + '\n')
