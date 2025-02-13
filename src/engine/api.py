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
import time

from PIL import Image
from singa import tensor
from singa import device
import numpy as np
from .config import get_dataset_config_from_dict
from .config import MLPConfig
from .config import CNNConfig
from .config import get_model_config_from_dict
from .config import TrainConfig
from .config import get_train_config_from_dict
from .config import SgdConfig
from .config import get_optimizer_config_from_dict
from .config import get_reg_config_from_dict

np_dtype = {"float16": np.float16, "float32": np.float32}
singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


def train(model_cfg, data_cfg, train_cfg, reg_cfg, opt_cfg):
    train_config = _get_train_config(train_cfg)
    # 训练配置
    max_epoch = train_config.max_epoch
    batch_size = train_config.batch_size
    precision = train_config.precision

    # 配置设备
    dev_config = train_config.device
    dev = _get_dev_by_config(dev_config)

    # 配置随机种子
    random_seed = train_config.random_seed
    dev.SetRandSeed(random_seed)
    np.random.seed(random_seed)

    # 获取正则化配置
    reg_config = _get_reg_config(reg_cfg)
    # 临时配置
    global_rank = 0
    world_size = 1
    local_rank = 0
    graph = False
    verbosity = 0
    dist_option='plain'
    spars = None

    data_config = _get_dataset_config(data_cfg)
    if data_config.name == "mnist":
        from data import mnist
        train_x, train_y, val_x, val_y = mnist.load(data_config.dir_path)
    else:
        raise ValueError(f"`r`Not support dataset {data_config.name}")

    data_num_channels = train_x.shape[1]
    image_size = train_x.shape[2]
    data_size = np.prod(train_x.shape[1:train_x.ndim]).item()
    data_num_classes = (np.max(train_y) + 1).item()

    print(data_num_channels, image_size, data_size, data_num_classes)

    # 获取模型
    mod = _get_model_by_config(model_cfg)  # raise value error

    if mod.dimension == 4:
        tx = tensor.Tensor(
            (batch_size, data_num_channels, mod.input_size, mod.input_size), dev, singa_dtype[precision])
    elif mod.dimension == 2:
        tx = tensor.Tensor((batch_size, data_size), dev, singa_dtype[precision])
        np.reshape(train_x, (train_x.shape[0], -1))
        np.reshape(val_x, (val_x.shape[0], -1))

    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    total_train = train_x.shape[0]
    num_train_batch = total_train // batch_size
    total_val = val_x.shape[0]
    num_val_batch = total_val // batch_size
    idx = np.arange(total_train, dtype=np.int32)

    # 配置优化器
    optimizer = _get_optimizer_by_config(opt_cfg)
    mod.set_optimizer(optimizer)

    mod.compile([tx], is_train=True, use_graph=False, sequential=False)
    dev.SetVerbosity(verbosity)

    for epoch in range(max_epoch):
        start_time = time.time()
        np.random.shuffle(idx)
        if global_rank == 0:
            print('Starting Epoch %d:' % (epoch))
        # Training phase
        train_correct = np.zeros(shape=[1], dtype=np.float32)
        test_correct = np.zeros(shape=[1], dtype=np.float32)
        train_loss = np.zeros(shape=[1], dtype=np.float32)

        mod.train()
        for b in range(num_train_batch):
            x = train_x[idx[b * batch_size:(b + 1) * batch_size]]
            if mod.dimension == 4:
                x = augmentation(x, batch_size)
                if image_size != mod.input_size:
                    x = resize_dataset(x, mod.input_size)
            x = x.astype(np_dtype[precision])
            y = train_y[idx[b * batch_size:(b + 1) * batch_size]]
            # Copy the patch data into input tensors
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            # Train the model
            out, loss = mod(tx, ty, reg_config, dist_option, spars)
            train_correct += accuracy(tensor.to_numpy(out), y)
            train_loss += tensor.to_numpy(loss)[0]

        if global_rank == 0:
            print('Training loss = %.2f, training accuracy = %.2f %%' %
                  (train_loss, train_correct / (total_train * world_size) * 100.0), flush=True)

        # Evaluation phase
        mod.eval()
        for b in range(num_val_batch):
            x = val_x[b * batch_size:(b + 1) * batch_size]
            if mod.dimension == 4:
                if image_size != mod.input_size:
                    x = resize_dataset(x, mod.input_size)
            x = x.astype(np_dtype[precision])
            y = val_y[b * batch_size:(b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            out_test = mod(tx)
            test_correct += accuracy(tensor.to_numpy(out_test), y)
        # Output the evaluation accuracy
        if global_rank == 0:
            print('Evaluation accuracy = %.2f %%, Elapsed Time = %fs' %
                  (test_correct / (total_val * world_size) * 100.0, time.time() - start_time), flush=True)


def _get_model_by_config(model_cfg):
    config_instance = get_model_config_from_dict(model_cfg) # raise value error
    if isinstance(config_instance, MLPConfig):
        from model import MLP
        in_features = config_instance.in_features
        hidden_features = config_instance.hidden_features
        out_features = config_instance.out_features
        mod = MLP(in_features=in_features, perceptron_size=hidden_features, num_classes=out_features)
        return mod

    if isinstance(config_instance, CNNConfig):
        from model import CNN
        in_channels = config_instance.in_channels
        out_channels = config_instance.out_channels
        mod = CNN(num_classes=out_channels, num_channels=in_channels)
        return mod

def _get_dev_by_config(dev_cfg):
    if dev_cfg == "cpu":
        return device.create_cpu_device()
    elif dev_cfg == "gpu":
        return device.create_cuda_gpu()
    else:
        return device.get_default_device()

def _get_optimizer_by_config(opt_cfg):
    config_instance = get_optimizer_config_from_dict(opt_cfg) # raise value error
    if isinstance(config_instance, SgdConfig):
        from singa import opt
        lr = config_instance.lr
        momentum = config_instance.momentum
        weight_decay = config_instance.weight_decay
        precision = config_instance.precision
        singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}
        sgd = opt.SGD(lr=lr, momentum=momentum, weight_decay=weight_decay, dtype=singa_dtype[precision])
        return sgd


def _get_train_config(train_cfg) -> TrainConfig:
    return get_train_config_from_dict(train_cfg)

def _get_reg_config(reg_cfg):
    return get_reg_config_from_dict(reg_cfg)

def _get_dataset_config(data_cfg):
    return get_dataset_config_from_dict(data_cfg)


# Data augmentation
def augmentation(x, batch_size):
    xpad = np.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]], 'symmetric')
    for data_num in range(0, batch_size):
        offset = np.random.randint(8, size=2)
        x[data_num, :, :, :] = xpad[data_num, :,
                                    offset[0]:offset[0] + x.shape[2],
                                    offset[1]:offset[1] + x.shape[2]]
        if_flip = np.random.randint(2)
        if (if_flip):
            x[data_num, :, :, :] = x[data_num, :, :, ::-1]
    return x


def resize_dataset(x, image_size):
    num_data = x.shape[0]
    dim = x.shape[1]
    X = np.zeros(shape=(num_data, dim, image_size, image_size),
                 dtype=np.float32)
    for n in range(0, num_data):
        for d in range(0, dim):
            X[n, d, :, :] = np.array(Image.fromarray(x[n, d, :, :]).resize(
                (image_size, image_size), Image.BILINEAR),
                                     dtype=np.float32)
    return X

# Calculate accuracy
def accuracy(pred, target):
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    return correct


if __name__ == '__main__':
    model_cfg_dict = {"name": "mlp", "in_features": 784, "out_features": 10, "hidden_features": 100, "bias": True}
    data_cfg_dict = {"name": "mnist", "dir_path": "/tmp/mnist"}
    train_cfg_dict = {"max_epoch": 10, "batch_size": 16, "device": "cpu", "precision": "float32", "random_seed": 0}
    reg_cfg_dict = {"name": "L2", "alpha": 0.5}
    # reg_cfg_dict = None
    opt_cfg_dict = {"name": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 1e-5, "precision": "float32"}
    train(model_cfg_dict, data_cfg_dict, train_cfg_dict, reg_cfg_dict, opt_cfg_dict)
