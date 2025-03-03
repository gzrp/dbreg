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

from sanic import Sanic
from sanic.exceptions import InvalidUsage
from sanic.response import json
from common.logger import get_logger
from trainer import TrainerBuilder

app = Sanic("train-app")

logger = get_logger("engine-app", "log")

@app.get("/health")
async def hello(request):
    return {"message": "hello engine-app."}

@app.post("/train")
async def train(request):
    # Check if request is JSON
    if not request.json:
        logger.info("Expecting JSON payload")
        raise InvalidUsage("Expecting JSON payload")
    json_request = request.json
    mdict = json_request.get("mdict")
    odict = json_request.get("odict")
    ddict = json_request.get("ddict")
    vdict = json_request.get("vdict")
    tdict = json_request.get("tdict")

    builder = TrainerBuilder()
    trainer = (builder.build_model(mdict)
               .build_optimizer(odict)
               .build_train_dataloader(ddict)
               .build_val_dataloader(vdict)
               .build_train_config(tdict)
               .build())

    task_id = trainer.tid

    logger.info("task_id: %s \n - ddict: %s \n - odict: %s \n - ddict: %s \n vdict: %s \n tdict: %s\n", task_id, ddict, odict, ddict, vdict, tdict)

    try:
       resp = trainer.train()
    except Exception as e:
        logger.error(e)
        return json({"error": "the training task failed to submit, specifically because: " + str(e)}, status=500)

    return json({"code": 200, "task_id": task_id, "result": resp})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8093)