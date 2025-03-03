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

import os
import logging
import traceback
import json
import orjson
import requests
from logging.handlers import TimedRotatingFileHandler

def get_logger(name, folder_name):
    if not os.path.exists(f"/tmp/{folder_name}"):
        os.makedirs(f"/tmp/{folder_name}")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    filename = f"/tmp/{folder_name}/{name}.log"
    fh = TimedRotatingFileHandler(filename, when='D', backupCount=7)
    sh = logging.StreamHandler()

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def exception_catcher(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error({"exception_func": func.__name__, "exception_msg": e, "traceback": traceback.format_exc()})
            return orjson.dumps(
                {"exception_func": func.__name__, "exception_msg": str(e)},
            ).decode('utf-8')
    return wrapper


logger = get_logger("pg-interface", "log")

@exception_catcher
def echo_python(msg: str):
    return orjson.dumps(msg).decode('utf-8')

# select train_base('{"name": "mlp", "in_features":10, "out_features":2, "hidden_features":16, "bias": true}','{"name": "sgd", "lr":0.01, "momentum":0.9, "weight_decay":0.00001, "precision": "float32"}','{"type":"stream", "svc_url": "http://192.168.56.20:8094", "table_name": "frappe_train", "namespace": "train", "columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"], "batch_size": 16}','{"device": "cpu", "seed":0, "max_epoch":3}','{"type":"stream", "svc_url": "http://192.168.56.20:8094", "table_name": "frappe_test", "namespace": "test", "columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"], "batch_size": 16}')

@exception_catcher
def train(encoded_str: str):
    print(encoded_str)
    params = json.loads(encoded_str)
    logger.info(params)
    mdict = params.get("mdict")
    if mdict is None:
        raise ValueError("mdict is None")
    odict = params.get("odict")
    if odict is None:
        raise ValueError("odict is None")

    ddict = params.get("ddict")
    if ddict is None:
        raise ValueError("ddict is None")
    tdict = params.get("tdict")
    if tdict is None:
        raise ValueError("tdict is None")
    vdict = params.get("vdict")
    if vdict is None:
        raise ValueError("vdict is None")
    args = {"mdict": mdict, "odict": odict, "ddict": ddict, "tdict": tdict,
            "vdict": vdict}
    logger.info(args)
    resp = requests.post('http://127.0.0.1:8093/train', json=args)
    return orjson.dumps(resp.json()).decode('utf-8')


# @exception_catcher
# def train_result(encoded_str: str):
#     params = json.loads(encoded_str)
#     logger.info(params)
#     task_id = params.get("task_id")
#     resp = requests.get('http://127.0.0.1:8000/results/{task_id}'.format(task_id=task_id))
#     return orjson.dumps(resp.json()).decode('utf-8')

