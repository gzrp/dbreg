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

import traceback
from datetime import datetime
from orjson import orjson


def exception_catcher(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            sign = '=' * 80 + '\n'
            print(f'{sign}>>>exception time: \t{datetime.now()}\n>>>exception func: \t{func.__name__}\n>>>exception msg: \t{e}')
            print(f'{sign}{traceback.format_exc()}{sign}')
    return wrapper

def exception_catcher_with_logger(logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                return orjson.dumps(
                    {"exception_time": datetime.now(), "exception_func": func.__name__, "exception_msg": e, "traceback": traceback.format_exc()},
                ).decode('utf-8')
        return wrapper
    return decorator
