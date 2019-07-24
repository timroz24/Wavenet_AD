import copy

import numpy as np
import pandas as pd
import itertools
from operator import itemgetter
from sklearn.model_selection import train_test_split


class DataGenerator(object):
    
    def __init__(self, equip_id_dict):
        self.equip_id_keys = equip_id_dict
        self.length = len(equip_id_dict)
        self.idx = np.arange(self.length)     
        
    def shuffle(self):
        np.random.shuffle(self.idx)
        
    def itemgetter(self,is_train = True):
        if is_train:
            return itemgetter(*self.train_idx)(self.equip_id_keys)
        else:
            return itemgetter(*self.test_idx)(self.equip_id_keys)
        
    def train_test_split(self, data_dict, train_size, random_state=np.random.randint(10000)):
        self.train_idx, self.test_idx = train_test_split(self.idx, train_size=train_size, random_state=random_state)        
#         self.train_ids, self.test_ids = itemgetter(*self.train_idx)(self.equip_id_keys), itemgetter(*self.test_idx)(self.equip_id_keys)
        self.train_length, self.test_length = len(self.train_idx), len(self.test_idx)
        print("train size", self.train_length, "val size", self.test_length)
#         train_dict, test_dict = itemgetter(*self.train_ids)(data_dict), itemgetter(*self.test_ids)(data_dict)
        

    def batch_generator(self, batch_size, batch_type, shuffle=True, num_epochs=10000, allow_smaller_final_batch=False):
        epoch_num = 0
        while epoch_num < num_epochs:
            if batch_type=='Train':
                if shuffle:
                    np.random.shuffle(self.train_idx)
                train_ids = self.itemgetter(is_train=True)
                for i in range(0, self.train_length + 1, batch_size):
                    batch_idx = train_ids[i: i + batch_size]
                    if not allow_smaller_final_batch and len(batch_idx) != batch_size:
                        break
                    yield batch_idx
            if batch_type =='Val':
                if shuffle:
                    np.random.shuffle(self.test_idx)
                test_ids = self.itemgetter(is_train=False)
                for i in range(0, self.test_length + 1, batch_size):
                    batch_idx = test_ids[i: i + batch_size]
                    if not allow_smaller_final_batch and len(batch_idx) != batch_size:
                        break
                    yield batch_idx
            epoch_num += 1
            
    def iterrows(self):
        for i in self.idx:
            yield self[i]

    def mask(self, mask):
        return DataFrame(copy.copy(self.columns), [mat[mask] for mat in self.data])

    def __iter__(self):
        return self.dict.items().__iter__()

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.dict[key]

        elif isinstance(key, int):
            return pd.Series(dict(zip(self.columns, [mat[self.idx[key]] for mat in self.data])))

    def __setitem__(self, key, value):
        assert value.shape[0] == len(self), 'matrix first dimension does not match'
        if key not in self.columns:
            self.columns.append(key)
            self.data.append(value)
        self.dict[key] = value
