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
import uvicorn
from exceptiongroup import catch
from fastapi import FastAPI
from fastapi import Request
from fastapi import HTTPException
from concurrent.futures import ProcessPoolExecutor
from common.logger import get_logger
from engine import api

app = FastAPI()

MAX_WORKERS = 1

executor = ProcessPoolExecutor(max_workers=2)

logger = get_logger("engine-app", "log")

@app.get("/hello")
async def hello():
    return {"message": "hello world."}

@app.post("/train")
async def train(request: Request):
    try:
        json_request = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON format")

    model_cfg = json_request.get("model_cfg")
    data_cfg = json_request.get("data_cfg")
    train_cfg = json_request.get("train_cfg")
    reg_cfg = json_request.get("reg_cfg")
    opt_cfg = json_request.get("opt_cfg")
    task_id = uuid.uuid1().hex
    json_request["task_id"] = task_id
    if len(executor._pending_work_items) >= MAX_WORKERS:
        return {"code": 500, "message": "the number of training tasks exceeds the maximum configured number. Please try again later."}

    try:
        executor.submit(api.train, task_id, model_cfg, data_cfg, train_cfg, reg_cfg, opt_cfg)
    except Exception as e:
        return {"code": 500, "message": "the training task failed to submit, specifically because: " + str(e)}

    return {"code": 200, "task_id": task_id, "message": "the training task has been submitted successfully! Please wait a few minutes to obtain the training records according to the task id."}


@app.get("/results/{task_id}")
async def result(task_id: str):
    try:
        ans = api.get_train_result(task_id)
        if ans is not None:
            ans_dict = ast.literal_eval(ans)
            return {"code": 200, "task_id": task_id, "result": ans_dict}
        else:
            return {"code": 200, "task_id": "task_id", "result": "the training task has not yet completed, please try again later."}
    except Exception as e:
        return {"code": 500, "task_id": task_id, "message": "failed to obtain training results. The reasons for the failure are as follows" + str(e)}



if __name__ == '__main__':

    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("done.")