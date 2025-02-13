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
from sanic.response import json
from sanic.exceptions import InvalidUsage

from common.logger import get_logger
from engine import api
app = Sanic("engine-app")

logger = get_logger("engine-app", "log")

@app.route("/hello")
async def hello(request):
    return json({"message": "hello world."})

@app.route("/train", methods=["POST"])
async def train(request):
    if not request.json:
        logger.info("Expecting json payload")
        raise InvalidUsage("Expecting json payload")

    json_request = request.json

    model_cfg = json_request.get("model_cfg")
    data_cfg = json_request.get("data_cfg")
    train_cfg = json_request.get("train_cfg")
    reg_cfg = json_request.get("reg_cfg")
    opt_cfg = json_request.get("opt_cfg")
    api.train(model_cfg, data_cfg, train_cfg, reg_cfg, opt_cfg)
    return json({"message": "hello world."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
