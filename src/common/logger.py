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
from logging.handlers import TimedRotatingFileHandler


def get_logger(name, folder_name):
    if not os.path.exists(f"./{folder_name}"):
        os.makedirs(f"./{folder_name}")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    filename = f"./{folder_name}/{name}.log"
    fh = TimedRotatingFileHandler(filename, when='D', backupCount=7)
    sh = logging.StreamHandler()

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

if __name__ == '__main__':
    logger1 = get_logger("logger1", "log1")
    logger1.info("logger1.info")
    logger1.debug("logger1.debug")
    logger1.warning("logger1.warning")
    logger1.error("logger1.error")
    logger1.critical("logger1.critical")

    logger2 = get_logger("logger2", "log2")
    logger2.info("logger2.info")
    logger2.debug("logger2.debug")
    logger2.warning("logger2.warning")
    logger2.error("logger2.error")
    logger2.critical("logger2.critical")


