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

from typing import Dict, Any
import numpy as np
from singa import model
from singa import  tensor
from singa import layer
from singa import autograd


np_dtype = {"float16": np.float16, "float32": np.float32}
singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


class MLP(model.Model):
    def __init__(self, in_features=10, hidden_features=100, out_features=10, bias=True):
        super(MLP, self).__init__()
        self.reg = None
        self.dimension = 2
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_classes = out_features
        self.relu = layer.ReLU()
        self.linear1 = layer.Linear(self.in_features, self.hidden_features, bias=bias)
        self.linear2 = layer.Linear(self.hidden_features, self.num_classes, bias=bias)
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()

    def forward(self, inputs):
        y = self.linear1(inputs)
        y = self.relu(y)
        y = self.linear2(y)
        return y

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)
        if self.reg is not None:
            name = self.reg.get("name")
            if name == "L2":
                alpha = self.reg.get("alpha")
                from .reg_loss import l2_loss_for_model
                reg_loss = l2_loss_for_model(self, alpha)
                loss = autograd.add(loss, reg_loss)

        if dist_option == 'plain':
            self.optimizer(loss)
        elif dist_option == 'half':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)
        return out, loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_reg(self, rdict):
        self.reg = rdict

# mdict{"name": "mlp", "in_features":10, "out_features":2, "hidden_features":16, "bias": true}
def create_mlp(mdict: Dict[str, Any]):
    in_features = mdict.get("in_features")
    if in_features is None:
        raise ValueError("mlp: in_features is not defined")

    if isinstance(in_features, int):
        if in_features <= 0:
            raise ValueError("mlp: in_features must be greater than 0")
    else:
        raise ValueError("mlp: in_features is not an integer")

    out_features = mdict.get("out_features")
    if out_features is None:
        raise ValueError("mlp: out_features is not defined")

    if isinstance(out_features, int):
        if out_features <= 0:
            raise ValueError("mlp: out_features must be greater than 0")
    else:
        raise ValueError("mlp: out_features is not an integer")

    hidden_features = mdict.get("hidden_features")
    if hidden_features is None:
        raise ValueError("mlp: hidden_features is not defined")

    if isinstance(hidden_features, int):
        if hidden_features <= 0:
            raise ValueError("mlp: hidden_features must be greater than 0")
    else:
        raise ValueError("mlp: hidden_features is not an integer")


    bias = mdict.get("bias")
    if bias is None:
        raise ValueError("mlp: bias is not defined")

    if not isinstance(bias, bool):
       raise ValueError("mlp: bias is not a boolean")

    rdict = mdict.get("reg")
    if rdict is not None:
        name = rdict.get("name")
        if name is None:
            raise ValueError("mlp: reg is not defined")

        alpha = rdict.get("alpha")
        if alpha is None:
            raise ValueError("mlp: alpha is not defined")

        if isinstance(alpha, str):
            alpha = float(alpha)

        if not isinstance(alpha, float):
            raise ValueError("mlp: alpha is not a float")

    mlp = MLP(in_features=in_features, hidden_features=hidden_features, out_features=out_features, bias=bias)
    mlp.set_reg(rdict)
    return mlp

__all__ = ['MLP', 'create_mlp']

if __name__ == '__main__':
    create_mlp({'in_features': 10, 'hidden_features': 10, 'out_features': 10, 'bias': True})