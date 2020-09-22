import numpy as np
from torch.utils.data import Dataset
import pickle


class RecommendationDataset(Dataset):
    """Recommendation Dataset."""
    def __init__(self, pkl_file, centroid_pkl_file, feature_mtx_pkl_file, device='cpu', return_item_id=False, return_item_info=False):
        """
        Args:
            pkl_file (string): Path to the csv file with annotations.
        """
        self.data = pickle.load(open(pkl_file, 'rb'))
        self.fea_mtx = pickle.load(open(feature_mtx_pkl_file, 'rb'))
        self.centroids_dict = pickle.load(open(centroid_pkl_file, 'rb'))
        self.device = device
        self.return_item_id = return_item_id
        self.return_item_info = return_item_info
    #
    def __len__(self):
        return len(self.data)
    #
    def __getitem__(self, idx):
        x = self.fea_mtx[self.data[idx]['x_ind']].toarray()
        session_id = np.int64(self.data[idx]['session_id'])
        y = np.array([self.data[idx]['y']], dtype=np.int64)
        if self.return_item_info:
            sample = {'x': x.flatten(), 'y': y.flatten(), 'session_id': np.array([session_id]), 'item_id': np.array([self.data[idx]['item_id']])}
            sample['pos_centroid'] = self.centroids_dict[session_id]['positive'].flatten()
            sample['neg_centroid'] = self.centroids_dict[session_id]['negative'].flatten()
        elif self.return_item_id:
            sample = {'x':  x.flatten(), 'y': y.flatten(), 'session_id': np.array([session_id]), 'item_id': np.array([self.data[idx]['item_id']])}
            sample['pos_centroid'] = self.centroids_dict[session_id]['positive'].flatten()
            sample['neg_centroid'] = self.centroids_dict[session_id]['negative'].flatten()
        else:
            sample = {'x': x.flatten(), 'y':y.flatten()}
            sample['pos_centroid'] = self.centroids_dict[session_id]['positive'].flatten()
            sample['neg_centroid'] = self.centroids_dict[session_id]['negative'].flatten()
        return sample


class RecommendationDatasetPairwise(Dataset):
    """
    Recommendation Dataset.
    """
    def __init__(self, pkl_file, feature_mtx_pkl_file, device='cpu', return_item_id=False, train=False):
        """
        Args:
            pkl_file (string): Path to the csv file with annotations.
        """
        self.data = pickle.load(open(pkl_file, 'rb'))
        self.fea_mtx = pickle.load(open(feature_mtx_pkl_file, 'rb'))
        self.device = device
        self.return_item_id = return_item_id
        self.train = train
    #
    def __len__(self):
        return len(self.data) #number of sessions
    #
    def __getitem__(self, idx):
        if self.train:
            label = np.random.choice(['positive', 'negative'], 1)
            if len(self.data[idx]['clicked_item_inds']) > 0:
                pos_i = np.random.choice(self.data[idx]['clicked_item_inds'], 1)[0]
            else:
                pos_i = -1
            if len(self.data[idx]['non-clicked_item_inds']) > 0:
                neg_i = np.random.choice(self.data[idx]['non-clicked_item_inds'], 1)[0]
            else:
                neg_i = -1
            if label == 'positive':
                str_ij = "{}_{}".format(pos_i, neg_i)
                if len(self.data[idx]['clicked_item_inds']) > 0:
                    x_i = self.fea_mtx[pos_i].toarray()
                else:
                    x_i = np.zeros_like(self.fea_mtx[0].toarray())
                if len(self.data[idx]['non-clicked_item_inds']) > 0:
                    x_j = self.fea_mtx[neg_i].toarray()
                else:
                    x_j = np.zeros_like(x_i)
                y_ij = np.array([1])
            else:
                str_ij = "{}_{}".format(neg_i, pos_i)
                if len(self.data[idx]['clicked_item_inds']) > 0:
                    x_j = self.fea_mtx[pos_i].toarray()
                else:
                    x_j = np.zeros_like(self.fea_mtx[0].toarray())
                if len(self.data[idx]['non-clicked_item_inds']) > 0:
                    x_i = self.fea_mtx[neg_i].toarray()
                else:
                    x_i = np.zeros_like(x_j)
                y_ij = np.array([0])
        if self.return_item_id:
            sample = {'x_i': x_i.flatten(), 'x_j': x_j.flatten(), 'y': y_ij, 'session_id': np.array([self.data[idx]['session_id']], dtype=np.int64), 'item_pair_ij':[str_ij]}
            sample['pos_centroid'] = self.data[idx]['pos_centroid'].flatten()
            sample['neg_centroid'] = self.data[idx]['neg_centroid'].flatten()
        else:
            sample = {'x_i': x_i.flatten(), 'x_j': x_j.flatten(), 'y': y_ij}
            sample['pos_centroid'] = self.data[idx]['pos_centroid'].flatten()
            sample['neg_centroid'] = self.data[idx]['neg_centroid'].flatten()
        return sample
