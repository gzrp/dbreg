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
import json
from typing import Dict, Any


class DataSetConfig:
    def __init__(self, name: str, total_size: int, train_size: int, val_size: int):
        self.name = name
        self.total_size = total_size
        self.train_size = train_size
        self.val_size = val_size

def get_dateset_config_from_json(config: str):
    def object_hook(cfg_dict):
        name = cfg_dict.get("name")
        total_size = cfg_dict.get("total_size")
        train_size = cfg_dict.get("train_size")
        val_size = cfg_dict.get("val_size")
        return DataSetConfig(name, total_size, train_size, val_size)
    return json.loads(config, object_hook=object_hook)

def get_dataset_config_from_dict(config: Dict[str, Any]) -> DataSetConfig:
    name = config.get("name")
    total_size = config.get("total_size")
    train_size = config.get("train_size")
    val_size = config.get("val_size")
    return DataSetConfig(name, total_size, train_size, val_size)

