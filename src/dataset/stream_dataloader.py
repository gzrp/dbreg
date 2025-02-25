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
                    self.data_queue.put({self.eos_signal: True})
                else:
                    id_npy = np.asarray(batch['id'], dtype=np.int32)
                    value_npy = np.asarray(batch['value'], dtype=np.float32)
                    y_npy = np.asarray(batch['y'], dtype=np.int32)

                    data_npy = {'id': id_npy, 'value': value_npy, 'y': y_npy}
                    self.data_queue.put(data_npy)
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


