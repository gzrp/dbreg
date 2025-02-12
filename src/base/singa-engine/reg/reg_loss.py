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

def l2_loss(model, alpha: float):
    loss = tensor.Tensor((1,), requires_grad=True).set_value(0.0)
    alpha_tensor = tensor.Tensor((1,), requires_grad=False, stores_grad=False).set_value(alpha)
    params = model.get_params()
    for name, param in params.items():
        if '.W' in name:
            loss_item = autograd.mse_loss(param, tensor.zeros_like(param))
            loss_item = autograd.mul(loss_item, alpha_tensor)
            loss = autograd.add(loss, loss_item)
    return loss


