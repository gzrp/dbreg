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

class TrainConfig:
    def __init__(self, max_epoch: int, batch_size: int, device: str = "cpu", precision: str ="float32"):
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.device = device
        self.precision = precision


def get_train_config_from_dict(config: Dict[str, Any]):
    max_epoch = config.get("max_epoch")
    batch_size = config.get("batch_size")
    device = config.get("device")
    precision = config.get("precision")
    return TrainConfig(max_epoch=max_epoch, batch_size=batch_size, device=device, precision=precision)
