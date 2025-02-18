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

import ast
import uuid
from sanic import Sanic
from sanic.exceptions import InvalidUsage
from sanic.response import json
from concurrent.futures import ProcessPoolExecutor
from common.logger import get_logger
from engine import api

app = Sanic("train-app")

MAX_WORKERS = 1

executor = ProcessPoolExecutor(max_workers=2)

logger = get_logger("engine-app", "log")

@app.get("/hello")
async def hello(request):
    return {"message": "hello engine-app."}

@app.post("/train")
async def train(request):
    # Check if request is JSON
    if not request.json:
        logger.info("Expecting JSON payload")
        raise InvalidUsage("Expecting JSON payload")
    json_request = request.json
    model_cfg = json_request.get("model_cfg")
    data_cfg = json_request.get("data_cfg")
    train_cfg = json_request.get("train_cfg")
    reg_cfg = json_request.get("reg_cfg")
    opt_cfg = json_request.get("opt_cfg")
    task_id = uuid.uuid1().hex
    json_request["task_id"] = task_id
    if len(executor._pending_work_items) >= MAX_WORKERS:
        return json({"error": "the number of training tasks exceeds the maximum configured number. Please try again later."}, status=500)

    try:
        executor.submit(api.train, task_id, model_cfg, data_cfg, train_cfg, reg_cfg, opt_cfg)
    except Exception as e:
        return json({"error": "the training task failed to submit, specifically because: " + str(e)}, status=500)

    return json({"code": 200, "task_id": task_id, "message": "the training task has been submitted successfully! Please wait a few minutes to obtain the training records according to the task id."})


@app.get("/results/<task_id>")
async def result(request, task_id: str):
    try:
        ans = api.get_train_result(task_id)
        if ans is not None:
            ans_dict = ast.literal_eval(ans)
            return json({"code": 200, "task_id": task_id, "result": ans_dict})
        else:
            return json({"code": 200, "task_id": task_id, "result": "the training task has not yet completed, please try again later."})
    except Exception as e:
        return json({"task_id": task_id, "message": "failed to obtain training results. The reasons for the failure are as follows" + str(e)}, status=500)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8093)