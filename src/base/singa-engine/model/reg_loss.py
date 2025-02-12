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

from singa import tensor
from singa import autograd
from singa import singa_wrap as singa


def l2_loss_for_model(model, alpha: float):
    loss = tensor.Tensor((1,), requires_grad=True).set_value(0.0)
    params = model.get_params()
    for name, param in params.items():
        if '.W' in name:
            loss_item = l2_loss(param, alpha)
            loss = autograd.add(loss, loss_item)
    return loss


class L2LossError(autograd.Operator):

    def __init__(self, t, alpha):
        super(L2LossError, self).__init__()
        self.t = t.data
        self.alpha = alpha

    def forward(self, x):
        self.err = singa.__sub__(x, self.t)
        sqr = singa.Square(self.err)
        loss = singa.SumAll(sqr)
        self.n = 1
        for s in x.shape():
            self.n *= s
        loss /= self.n
        loss *= self.alpha
        return loss

    def backward(self, dy=1.0):
        dx = self.err
        dx *= float(2 / self.n)
        dx *= self.alpha
        dx *= dy
        return dx

def l2_loss(x, alpha):
    t = tensor.zeros_like(x)
    return L2LossError(t, alpha)(x)[0]