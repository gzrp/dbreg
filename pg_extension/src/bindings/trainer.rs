/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

use crate::bindings::register::PY_TRAIN_MODULE;
use crate::bindings::register::run_python_train_function;


pub fn echo_python(message: &String) -> serde_json::Value {
    run_python_train_function(&PY_TRAIN_MODULE, message, "echo_python")
}

pub fn train(args: &String) -> serde_json::Value {
    run_python_train_function(&PY_TRAIN_MODULE, args, "train")
}

pub fn train_result(args: &String) -> serde_json::Value {
    run_python_train_function(&PY_TRAIN_MODULE, args, "train_result")
}