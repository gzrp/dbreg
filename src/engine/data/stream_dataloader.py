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

import queue
import threading
import time
from typing import Dict, Any

import requests
import numpy as np

from common import get_logger

logger = get_logger("stream-dataloader", "log")

class StreamDataloader:
    def __init__(self, svc_url, table, namespace):
        self.svc_url = svc_url
        self.table = table
        self.namespace = namespace
        self.eos_signal = "eos"
        self.last_fetch_time = 0
        self.data_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._fetch_data, daemon=True)
        self.thread.start()

    def _fetch_data(self):
        while not self.stop_event.is_set():
            resp = requests.get(f'{self.svc_url}/get', params={'table_name': self.table, 'namespace': self.namespace})
            if resp.status_code == 200:
                batch = resp.json()
                if batch == self.eos_signal:
                    logger.info("[StreamingDataLoader] eos...")
                    self.data_queue.put({self.eos_signal: True}, block=True)
                else:
                    id_npy = np.asarray(batch['id'], dtype=np.float32)
                    value_npy = np.asarray(batch['value'], dtype=np.float32)
                    y_npy = np.asarray(batch['y'], dtype=np.int32)

                    data_npy = {'id': id_npy, 'value': value_npy, 'y': y_npy}
                    self.data_queue.put(data_npy, block=True)
            else:
                logger.error(resp.json())
                time.sleep(5)

    def __iter__(self):
        return self

    def __next__(self):
        # logger.info("next-spendtime:%f" % (time.time() - self.last_fetch_time))
        self.last_fetch_time = time.time()
        if not self.thread.is_alive():
            logger.error("stream-dataloader thread is dead")
            raise StopIteration
        else:
            data = self.data_queue.get(block=True)
            if self.eos_signal in data:
                raise StopIteration
            else:
                return data


    def __len__(self):
        return self.data_queue.qsize()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

# ddict{"type":"stream", "svc_url": "http://192.168.56.20:8094", "table_name": "frappe_train", "namespace": "train", "columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"], "batch_size": 16}
def create_stream_dataloader(ddict: Dict[str, Any]):
    logger.info("ddict: %s", ddict)
    svc_url = ddict.get("svc_url")
    if svc_url is None:
        raise ValueError("stream: svc_url is not defined")

    # cache web service health check
    resp = requests.get(f'{svc_url}/health')
    if resp.status_code != 200:
        raise ValueError("stream: svc_url is not healthy")

    table_name = ddict.get("table_name")
    if table_name is None:
        raise ValueError("stream: table_name is not defined")

    namespace = ddict.get("namespace")
    if namespace is None:
        raise ValueError("stream: namespace is not defined")

    columns = ddict.get("columns")
    if columns is None:
        raise ValueError("stream: columns is not defined")

    if not isinstance(columns, list):
        raise ValueError("stream: columns is not a list")

    batch_size = ddict.get("batch_size")
    if batch_size is None:
        raise ValueError("stream: batch_size is not defined")

    if not isinstance(batch_size, int):
        raise ValueError("stream: batch_size is not an integer")

    if batch_size <= 0:
        raise ValueError("stream: batch_size is not positive")

    # start the cache service
    rdict = {"table_name": table_name, "namespace": namespace, "columns": columns, "batch_size": batch_size}
    resp = requests.post(f'{svc_url}/start', json=rdict)

    if resp.status_code != 200:
        raise ValueError("stream: cannot start stream")

    time.sleep(0.2)

    loader = StreamDataloader(svc_url, table_name, namespace)
    return loader


__all__ = ["create_stream_dataloader"]

