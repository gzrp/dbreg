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
    def __init__(self, conn_cfg: ConnConfig, table: str, namespace: str, columns: List, batch_size: int, max_size: int = 128):
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
                    self.queue.put(batch, block=True)
                    logger.info(f"table={self.table} data is fetched, queue_size={self.queue.qsize()}, time_usg={time_usg}")
                    # block until a free slot is available
                    time.sleep(0.001)
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
            logger.info(f"table={self.table} data is fetched eos")
            return "eos", time.time() - begin_time

        # print("last_id", self.table, self.last_id)

        batch = self._preprocess(rows)
        batch["last_id"] = self.last_id
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

        # 如果不够 batch size, 则使用最后一个进行填充
        pad_size = self.batch_size - sample_lines
        for i in range(pad_size):
            feat_id.append(feat_id[-1])
            feat_value.append(feat_value[-1])
            y.append(y[-1])

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
        return self.queue.get(block=True)

    def empty(self):
        return self.queue.empty()

    def size(self):
        return self.queue.qsize()


app = Sanic("cache-app")

@app.get("/health")
async def hello(request):
    return json({"code": 200, "message": "the cache service is health."})


@app.post("/start")
async def start_cache(request):
    logger.info("start cache, request: %s", request.json)
    # Check if request is JSON
    if not request.json:
        logger.info("Expecting JSON payload")
        raise InvalidUsage("Expecting JSON payload")

    json_request = request.json
    columns = json_request.get("columns")
    namespace = json_request.get("namespace")
    if columns is None:
        return json({"error": "No columns specified"}, status=500)
    if namespace not in ["train", "valid", "test"]:
        return json({"error": "namespace is not correct"}, status=500)

    table_name = json_request.get("table_name")
    batch_size = json_request.get("batch_size")

    try:
        if not hasattr(app.ctx, f'{table_name}_{namespace}_cache'):
            setattr(app.ctx, f'{table_name}_{namespace}_cache', CacheService(db_conn, table_name, namespace, columns, batch_size))
        return json({"code": 200, "message": "the cache service is started successfully."})
    except Exception as e:
        return json({"error": str(e)}, status=500)


@app.get("/get")
async def get(request):
    namespace = request.args.get("namespace")
    table_name = request.args.get("table_name")
    # check if exist
    if not hasattr(app.ctx, f'{table_name}_{namespace}_cache'):
        logger.info(f"{table_name}_{namespace}_cache not start yet")
        return json({"error": f"{table_name}_{namespace}_cache not start yet"}, status=404)

    batch_data = getattr(app.ctx, f'{table_name}_{namespace}_cache').get()
    # return
    if batch_data is None:
        logger.info("error: No data available")
        return json({"error": "No data available"}, status=404)
    else:
        return json(batch_data)


@app.post("/remove")
async def remove(request):
    namespace = request.args.get("namespace")
    table_name = request.args.get("table_name")
    # check if exist
    if hasattr(app.ctx, f'{table_name}_{namespace}_cache'):
        delattr(app.ctx, f'{table_name}_{namespace}_cache')
    return json({"code": 200, "message": "the cache service is removed successfully."}, status=200)




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8094)
