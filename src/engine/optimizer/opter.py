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
from singa import opt
from singa import tensor

singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


def create_opt(odict: Dict[str, Any]):
    name = odict["name"]
    if name == "sgd":
        o = create_sgd(odict)
        return o
    else:
        raise ValueError(f"{name} optimizer is not support")


# odict{"name": "sgd", "lr":0.01, "momentum":0.9, "weight_decay":0.00001, "precision": "float32"}
def create_sgd(odict: Dict[str, Any]):
    lr = odict.get("lr")
    if lr is None:
        raise ValueError("sgd: lr is not defined")

    if isinstance(lr, float):
        if lr <= 0:
            raise ValueError("sgd: lr must be positive")
    else:
        raise ValueError("sgd: lr is not a float")

    momentum = odict.get("momentum")
    if momentum is None:
        raise ValueError("sgd: momentum is not defined")

    if isinstance(momentum, float):
        if momentum <= 0:
            raise ValueError("sgd: momentum must be positive")
    else:
        raise ValueError("sgd: momentum is not a float")

    weight_decay = odict.get("weight_decay")
    if weight_decay is None:
        raise ValueError("sgd: weight_decay is not defined")

    if isinstance(weight_decay, float):
        if weight_decay <= 0:
            raise ValueError("sgd: weight_decay must be positive")
    else:
        raise ValueError("sgd: weight_decay is not a float")

    precision = odict.get("precision")
    if precision is None:
        raise ValueError("sgd: precision is not defined")

    if isinstance(precision, str):
        if precision != "float32" and precision != "float16":
            raise ValueError("sgd: precision must be float32 or float16")
    else:
        raise ValueError("sgd: precision is not a string")

    sgd = opt.SGD(lr=lr, momentum=momentum, weight_decay=weight_decay, dtype=singa_dtype[precision])
    return sgd

__all__ = ['create_sgd']

if __name__ == '__main__':
    create_sgd({"name": "sgd", "lr":0.01, "momentum":0.9, "weight_decay":0.00001, "precision": "float32"})
