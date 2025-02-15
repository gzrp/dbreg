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
import orjson
import requests
from common.catcher import exception_catcher_with_logger
from common.logger import get_logger

logger = get_logger("pg-interface", "log")


@exception_catcher_with_logger(logger=logger)
def echo_python(msg: str):
    return orjson.dumps(msg).decode('utf-8')


@exception_catcher_with_logger(logger=logger)
def train(encoded_str: str):
    params = json.loads(encoded_str)
    logger.info(params)

    model_cfg = params.get("model_cfg")
    data_cfg = params.get("data_cfg")
    train_cfg = params.get("train_cfg")
    reg_cfg = params.get("reg_cfg")
    opt_cfg = params.get("opt_cfg")

    reg_cfg["alpha"] = float(reg_cfg.get("alpha"))

    opt_cfg["momentum"] = float(opt_cfg.get("momentum"))
    opt_cfg["lr"] = float(opt_cfg.get("lr"))
    opt_cfg["weight_decay"] = float(opt_cfg.get("weight_decay"))

    logger.info(model_cfg)
    logger.info(data_cfg)
    logger.info(train_cfg)
    logger.info(reg_cfg)
    logger.info(opt_cfg)

    args = {"model_cfg": model_cfg, "data_cfg": data_cfg, "train_cfg": train_cfg, "reg_cfg": reg_cfg,
            "opt_cfg": opt_cfg}

    # return orjson.dumps(args).decode('utf-8')
    resp = requests.post('http://127.0.0.1:8000/train', json=args)
    return orjson.dumps(resp.json()).decode('utf-8')


@exception_catcher_with_logger(logger=logger)
def train(encoded_str: str):
    params = json.loads(encoded_str)
    logger.info(params)
    task_id = params.get("task_id")

    resp = requests.get('http://127.0.0.1:8000/results/{task_id}'.format(task_id=task_id))
    return orjson.dumps(resp.json()).decode('utf-8')



