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

from typing import Dict, Any, Union, List


class RegConfig:
    def __init__(self, name: str):
        self.name = name

class L2RegConfig(RegConfig):
    def __init__(self, name: str, alpha):
        super().__init__(name)
        self.alpha = alpha

class L1RegConfig(RegConfig):
    def __init__(self, name: str, beta):
        super().__init__(name)
        self.beta = beta

def get_reg_config_from_list_or_dict(config: Dict[str, Any]):
    name = config.get("name")
    if name == "L2":
        alpha = config.get("alpha")
        return L2RegConfig(name, alpha)
    elif name == "L1":
        beta = config.get("beta")
        return L1RegConfig(name, beta)
    else:
        raise ValueError(f"Reg {name} not supported.")





