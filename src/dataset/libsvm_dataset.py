#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def numpy_collate_fn(batch):
    return batch

class LibSvmDataset(Dataset):
    def __init__(self, file_path, fields):
        self.file_path = file_path
        self.samples_cnt = self._libsvm_lines_cnt(file_path)
        self.fields = fields
        self.feat_id =  np.zeros((self.samples_cnt, self.fields), dtype=np.int32)
        self.feat_value = np.zeros((self.samples_cnt, self.fields), dtype=np.float32)
        self.y = np.zeros(self.samples_cnt, dtype=np.float32)
        # load data
        self._libsvm_load()

    def _libsvm_load(self):
        idx = 0
        with tqdm(total=self.samples_cnt) as pbar:
            with open(self.file_path) as fp:
                line = fp.readline()
                while line:
                    sample = self._libsvm_decode(line)
                    self.feat_id[idx] = sample['id']
                    self.feat_value[idx] = sample['value']
                    self.y[idx] = sample['y']
                    idx += 1
                    line = fp.readline()
                    pbar.update(1)


    @staticmethod
    def _libsvm_decode(line):
        columns = line.split(' ')
        map_func = lambda pair: (int(pair[0]), float(pair[1]))
        id, value = zip(*map(lambda col: map_func(col.split(':')), columns[1:]))
        sample = {'id': list(id),
                  'value': list(value),
                  'y': float(columns[0])}
        return sample

    @staticmethod
    def _libsvm_lines_cnt(file_path):
        with open(file_path) as f:
            line_cnt = sum(1 for _ in f)
        return line_cnt

    def __len__(self):
        return self.samples_cnt

    def __getitem__(self, idx):
        return {'id': self.feat_id[idx],
                'value': self.feat_value[idx],
                'y': self.y[idx]}

def libsvm_dataloader(file_dir, fields, batch_size, collect_type=None, workers=1):
    train_path = os.path.join(file_dir, 'train.libsvm')
    valid_path = os.path.join(file_dir, 'valid.libsvm')
    test_path = os.path.join(file_dir, 'test.libsvm')

    collate_fn = None
    if collect_type == 'numpy':
        collate_fn = numpy_collate_fn

    train_loader = DataLoader(LibSvmDataset(train_path, fields), batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=workers)
    valid_loader = DataLoader(LibSvmDataset(valid_path, fields), batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=workers)
    test_loader = DataLoader(LibSvmDataset(test_path, fields), batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=workers)

    return train_loader, valid_loader, test_loader
