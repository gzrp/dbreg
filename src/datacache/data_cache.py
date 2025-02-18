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
from typing import List
import psycopg2
from sanic import Sanic
from sanic.response import json
from sanic.exceptions import InvalidUsage
from common.logger import get_logger

logger = get_logger("cache-app", "log")


class ConnConfig:
    def __init__(self, user, host, port, db):
        self.user = user
        self.host = host
        self.port = port
        self.db = db


USER = "zrp"
HOST = "127.0.0.1"
PORT = "28814"
DB_NAME = "pg_extension"
CACHE_SIZE = 10

db_conn = ConnConfig(USER, HOST, PORT, DB_NAME)

class CacheService:
    def __init__(self, conn_cfg: ConnConfig, table: str, namespace: str, columns: List, batch_size: int, max_size: int = 10):
        self.conn_cfg = conn_cfg
        self.table = table                  # dataset
        self.namespace = namespace          # train valid test
        self.columns = columns
        self.batch_size = batch_size
        self.max_size = max_size            # max batches
        self.last_id = -1

        self.queue = queue.Queue(maxsize=max_size)

        self.thread = threading.Thread(target=self._put_data, daemon=True)
        self.thread.start()

    def _put_data(self):
        with psycopg2.connect(database=self.conn_cfg.db, user=self.conn_cfg.user, host=self.conn_cfg.host, port=self.conn_cfg.port) as conn:
            while True:
                try:
                    batch, time_usg = self._fetch_and_preprocess(conn)
                    self.queue.put(batch)
                    logger.info(f"data is fetched, queue_size={self.queue.qsize()}, time_usg={time_usg}")
                    # block until a free slot is available
                    time.sleep(0.1)
                except psycopg2.OperationalError:
                    logger.exception("database connection failure, trying to reconnect...")
                    time.sleep(5)  # wait before trying to establish a new connection
                    conn = psycopg2.connect(database=self.conn_cfg.db, user=self.conn_cfg.user, host=self.conn_cfg.host, port=self.conn_cfg.port)

    def _fetch_and_preprocess(self, conn):
        begin_time = time.time()
        cur = conn.cursor()
        columns_str = ", ".join(self.columns)

        cur.execute(f"SELECT id, {columns_str} FROM {self.table} "
                    f"WHERE id > {self.last_id} ORDER BY id ASC LIMIT {self.batch_size}")
        rows = cur.fetchall()

        if rows:
            self.last_id = max(row[0] for row in rows)
        else:
            # reset last id
            self.last_id = -1
            return "eos", time.time() - begin_time

        batch = self._preprocess(rows)
        return batch, time.time() - begin_time

    def _preprocess(self, rows: List[tuple]):
        sample_lines = len(rows)
        feat_id = []
        feat_value = []
        y = []
        for i in range(sample_lines):
            row_value = rows[i]
            sample = self._libsvm_decode(row_value)
            feat_id.append(sample['id'])
            feat_value.append(sample['value'])
            y.append(sample['y'])
        return {'id': feat_id, 'value': feat_value, 'y': y}

    @staticmethod
    def _libsvm_decode(row):
        map_func = lambda pair: (int(pair[0]), float(pair[1]))
        # 0 is id, 1 is label
        id, value = zip(*map(lambda col: map_func(col.split(':')), row[2:]))
        sample = {'id': list(id),
                  'value': list(value),
                  'y': int(row[1])}
        return sample

    def get(self):
        return self.queue.get()

    def empty(self):
        return self.queue.empty()

    def size(self):
        return self.queue.qsize()


app = Sanic("cache-app")

@app.get("/hello")
async def hello(request):
    return {"message": "hello cache-app."}


@app.post("/start")
async def start_cache(request):
    # Check if request is JSON
    if not request.json:
        logger.info("Expecting JSON payload")
        raise InvalidUsage("Expecting JSON payload")

    json_request = request.json
    columns = json_request.get("columns")
    namespace = json_request.get("namespace")
    if columns is None:
        return json({"code": 500, "message": "No columns specified"})
    if namespace not in ["train", "valid", "test"]:
        return json({"code": 500, "message": "namespace is not correct"})

    table_name = json_request.get("table_name")
    batch_size = json_request.get("batch_size")

    try:
        if not hasattr(app.ctx, f'{table_name}_{namespace}_cache'):
            setattr(app.ctx, f'{table_name}_{namespace}_cache', CacheService(db_conn, table_name, namespace, columns, batch_size))
        return json({"code": 200, "message": "the cache service is started successfully."})
    except Exception as e:
        return json({"code": 500, "message": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8094)
