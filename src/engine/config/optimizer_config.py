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

class OptimizerConfig:
    def __init__(self, name: str, lr: float):
        self.name = name
        self.lr = lr

class SgdConfig(OptimizerConfig):
    def __init__(self, name: str, lr: float, momentum: float, weight_decay: float, precision: str = "float32"):
        super().__init__(name, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.precision = precision

def get_optimizer_config_from_dict(config: Dict[str, Any]):
    name = config.get("name")
    if name == "sgd":
        lr = config.get("lr")
        momentum = config.get("momentum")
        weight_decay = config.get("weight_decay")
        precision = config.get("precision")
        return SgdConfig(name, lr, momentum, weight_decay, precision)
    else:
        raise ValueError(f"optimizer: {name} is unsupported.")

