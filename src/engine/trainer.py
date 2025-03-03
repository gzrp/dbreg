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

import uuid
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from singa import device, tensor

from common import get_logger

logger = get_logger("train", "/log")

np_dtype = {"float16": np.float16, "float32": np.float32}
singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}

def accuracy(pred, target):
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    return correct

class Trainer:
    def __init__(self):
        self.__ddict = None
        self.__odict = None
        self.__mdict = None
        self.__vdict = None
        self.tid = None
        self.name = None
        self.opt = None
        self.model = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.acc_func = None

        self.batch_size = None
        self.val_batch_size = None
        self.dev = None
        self.seed = None
        self.max_epoch = None


    def train(self):

        res = {}
        # 为模型设置优化器
        self.model.set_optimizer(self.opt)
        # 设置随机种子
        np.random.seed(self.seed)
        # 训练
        train_records = []
        test_records = []
        for epoch in range(self.max_epoch):
            start_time = time.time()
            # Training phase
            train_correct = np.zeros(shape=[1], dtype=np.float32)
            train_loss = np.zeros(shape=[1], dtype=np.float32)

            # 训练模式
            self.model.train()
            total = 0
            for idx, batch in enumerate(self.train_dataloader, start=1):
                x = batch['id']
                y = batch['y']
                # print(x)
                tx = tensor.Tensor(x.shape, self.dev, singa_dtype['float32'])
                ty = tensor.Tensor((y.shape[0],), self.dev, tensor.int32)

                tx.copy_from_numpy(x)
                ty.copy_from_numpy(y)
                out, loss = self.model(tx, ty, dist_option='plain', spars=None)
                train_loss += tensor.to_numpy(loss)[0]
                train_correct += self.acc_func(tensor.to_numpy(out), y)
                total += y.shape[0]
            train_record = "epoch-%d: loss: %.6f, acc:%d/%d=%.2f%%" % (epoch, train_loss, train_correct, total, 100.0 * train_correct / total)
            train_records.append({"epoch": epoch, "loss": '%.6f' % train_loss, "acc": '%.2f%%' % (100.0 * train_correct / total)})
            logger.info(f"task_id: {self.tid} - train - {train_record}")

            # 验证模式
            self.model.eval()
            test_correct = np.zeros(shape=[1], dtype=np.float32)
            eval_total = 0
            print("-------------------------------------------")
            for idx, batch in enumerate(self.val_dataloader, start=1):
                x = batch['id']
                y = batch['y']
                # print(x)
                tx = tensor.Tensor(x.shape, self.dev, singa_dtype['float32'])
                ty = tensor.Tensor((y.shape[0],), self.dev, tensor.int32)

                tx.copy_from_numpy(x)
                ty.copy_from_numpy(y)

                out_test = self.model(tx)
                test_correct += self.acc_func(tensor.to_numpy(out_test), y)
                eval_total += y.shape[0]

            test_record = "epoch-%d: acc:%d/%d=%.2f%%" % (epoch, test_correct, eval_total, 100.0 * test_correct / eval_total)
            test_records.append({"epoch": epoch, "acc": '%.2f%%' % (100.0 * test_correct / eval_total)})
            logger.info(f"task_id: {self.tid} - test - {test_record}")
        res["train_records"] = train_records
        res["test_records"] = test_records
        logger.info(f"task_id: {self.tid}, result: {res}")
        return res

class BaseBuilder(ABC):
    @abstractmethod
    def build_model(self, mdict: Dict[str, Any]):
        pass

    @abstractmethod
    def build_optimizer(self, odict: Dict[str, Any]):
        pass

    @abstractmethod
    def build_train_dataloader(self, ddict: Dict[str, Any]):
        pass

    @abstractmethod
    def build_val_dataloader(self, vdict: Dict[str, Any]):
        pass

    @abstractmethod
    def build_base_config(self, tdict: Dict[str, Any]):
        pass

    @abstractmethod
    def build_acc_func(self, func):
        pass


class TrainerBuilder(BaseBuilder):
    def __init__(self):
        self.trainer = Trainer()
        self.trainer.tid = uuid.uuid1().hex

    def build_optimizer(self, odict: Dict[str, Any]):
        if odict is None:
            raise ValueError("odict is None")

        if "name" not in odict:
            raise ValueError("name is not in odict")

        from optimizer.opter import create_sgd
        o = create_sgd(odict)
        self.trainer.__odict = odict
        self.trainer.opt = o
        return self


    def build_train_dataloader(self, ddict: Dict[str, Any]):
        if ddict is None:
            raise ValueError("ddict is None")

        if "type" not in ddict:
            raise ValueError("type is not in ddict")

        if "batch_size" not in ddict:
            raise ValueError("batch_size is not in ddict")

        batch_size = ddict.get("batch_size")
        if not isinstance(batch_size, int):
            raise ValueError("batch_size is not int")

        if batch_size <= 0:
            raise ValueError("batch_size is not positive")

        self.trainer.batch_size = batch_size

        from data.loader import create_loader
        d = create_loader(ddict)
        self.trainer.__ddict = ddict
        self.trainer.train_dataloader = d
        return self

    def build_model(self, mdict: Dict[str, Any]):

        if mdict is None:
            raise ValueError("mdict is None")

        if "name" not in mdict:
            raise ValueError("name is not in mdict")

        from model.moder import create_model
        # from .model.moder import create_model
        m = create_model(mdict)
        self.trainer.__mdict = mdict
        self.trainer.model = m
        return self

    # {"device": "cpu", "seed":0, "max_epoch":3}
    def build_base_config(self, tdict: Dict[str, Any]):
        dev = tdict.get("device")
        if dev is None:
            raise ValueError("device is not in tdict")

        if dev != "cpu":
            raise ValueError(f"device is not support {dev}")

        self.trainer.dev = device.get_default_device()

        seed = tdict.get("seed")
        if self is None:
            raise ValueError("seed is not in tdict")

        if not isinstance(seed, int):
            raise ValueError("seed is not an integer")

        self.trainer.seed = seed

        self.trainer.dev.SetRandSeed(seed)

        max_epoch = tdict.get("max_epoch")
        if max_epoch is None:
            raise ValueError("max_epoch is not in tdict")

        if not isinstance(max_epoch, int):
            raise ValueError("max_epoch is not an integer")

        if max_epoch <= 0:
            raise ValueError("max_epoch is not a positive integer")

        self.trainer.max_epoch = max_epoch
        return self

    def build_val_dataloader(self, vdict: Dict[str, Any]):
        if vdict is None:
            raise ValueError("ddict is None")

        if "type" not in vdict:
            raise ValueError("type is not in vdict")

        if "batch_size" not in vdict:
            raise ValueError("batch_size is not in vdict")

        batch_size = vdict.get("batch_size")
        if not isinstance(batch_size, int):
            raise ValueError("batch_size is not int")

        if batch_size <= 0:
            raise ValueError("batch_size is not positive")

        self.trainer.val_batch_size = batch_size

        from data.loader import create_loader
        v = create_loader(vdict)
        self.trainer.__vdict = vdict
        self.trainer.val_dataloader = v
        return self

    def build_acc_func(self, f):
        if f is not None:
            self.trainer.acc_func = f
        else:
            self.trainer.acc_func = accuracy
        return self


    def build(self):
        if self.trainer.model is None:
            raise ValueError("model is None")

        if self.trainer.opt is None:
            raise ValueError("opt is None")

        if self.trainer.train_dataloader is None:
            raise ValueError("train_dataloader is None")

        if self.trainer.dev is None:
            raise ValueError("dev is None")

        if self.trainer.seed is None:
            raise ValueError("seed is None")

        if self.trainer.max_epoch is None:
            raise ValueError("max_epoch is None")

        if self.trainer.batch_size is None:
            raise ValueError("batch_size is None")

        if self.trainer.acc_func is None:
            self.trainer.acc_func = accuracy

        return self.trainer

__all__ = ["TrainerBuilder"]

if __name__ == '__main__':

    mdict = {"name": "mlp", "in_features":10, "out_features":2, "hidden_features":16, "bias": True}
    odict = {"name": "sgd", "lr":0.01, "momentum":0.9, "weight_decay":0.00001, "precision": "float32"}
    ddict = {"svc_url": "http://192.168.56.20:8094", "table_name": "frappe_train", "namespace": "train", "columns": ["label", "col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"], "batch_size": 16}
    vdict = {"svc_url": "http://192.168.56.20:8094", "table_name": "frappe_test", "namespace": "test", "columns": ["label", "col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"], "batch_size": 16}
    tdict = {"device": "cpu", "seed": 0, "max_epoch": 30}


    builder = TrainerBuilder()
    trainer = (builder.build_name("trainer1")
               .build_model(mdict)
               .build_optimizer(odict)
               .build_train_dataloader(ddict)
               .build_val_dataloader(vdict)
               .build_train_config(tdict)
               .build())

    trainer.train()
    print("done")

