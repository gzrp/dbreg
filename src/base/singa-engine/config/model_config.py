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


class ModelConfig:
    def __init__(self, name):
        self.name = name


class MLPConfig(ModelConfig):
    def __init__(self, name, in_features, out_features, hidden_features, bias=True):
        super(MLPConfig, self).__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.bias = bias

class CNNConfig(ModelConfig):
    def __init__(self, name, in_channels, out_channels, bias=True):
        super(CNNConfig, self).__init__(name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias


def get_model_config_from_dict(config: Dict[str, Any]):
    name = config.get("name")
    if name == "mlp":
        in_features = config.get("in_features")
        out_features = config.get("out_features")
        hidden_features = config.get("hidden_features")
        bias = config.get("bias", True)
        return MLPConfig("mlp", in_features, out_features, hidden_features, bias)
    elif name == "cnn":
        in_features = config.get("in_channels")
        out_features = config.get("out_channels")
        bias = config.get("bias", True)
        return CNNConfig("cnn", in_features, out_features, bias)
    else:
        raise ValueError(f"model: {name} is unsupported.")


