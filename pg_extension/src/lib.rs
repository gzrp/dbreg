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

use std::collections::HashMap;
use pgrx::prelude::*;
use serde_json::json;
use serde_json::Value;
::pgrx::pg_module_magic!();

pub mod bindings;


#[pg_extern]
fn hello_pg_extension() -> &'static str {
    "Hello, pg_extension111"
}


#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "echo_python")]
fn echo_python(message: String) -> String {
    let mut args_map = HashMap::new();
    args_map.insert("message", message);
    let args_json = json!(args_map).to_string();
    crate::bindings::trainer::echo_python(&args_json).to_string()
}

#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "train")]
fn train(model_name: String,
         model_in_features: i32,
         model_out_features: i32,
         model_hidden_features: i32,
         model_bias: bool,
         data_name: String,
         data_dir_path: String,
         train_max_epoch: i32,
         train_batch_size: i32,
         train_device: String,
         train_precision: String,
         train_random_seed: i32,
         reg_name: String,
         reg_alpha: String,
         opt_name: String,
         opt_lr: String,
         opt_momentum: String,
         opt_weight_decay: String,
         opt_precision: String
) -> String {
    let mut train_map = HashMap::new();
    // model config
    let mut model_config: HashMap<String, Value> = HashMap::new();
    model_config.insert("name".to_string(), Value::from(model_name));
    model_config.insert("in_features".to_string(), Value::from(model_in_features));
    model_config.insert("out_features".to_string(), Value::from(model_out_features));
    model_config.insert("hidden_features".to_string(), Value::from(model_hidden_features));
    model_config.insert("bias".to_string(), Value::from(model_bias));

    train_map.insert("model_cfg".to_string(), model_config);

    // data config
    let mut data_config: HashMap<String, Value> = HashMap::new();
    data_config.insert("name".to_string(), Value::from(data_name));
    data_config.insert("dir_path".to_string(), Value::from(data_dir_path));

    train_map.insert("data_cfg".to_string(), data_config);

    // train config
    let mut train_config = HashMap::new();
    train_config.insert("max_epoch".to_string(), Value::from(train_max_epoch));
    train_config.insert("batch_size".to_string(), Value::from(train_batch_size));
    train_config.insert("device".to_string(), Value::from(train_device));
    train_config.insert("precision".to_string(), Value::from(train_precision));
    train_config.insert("random_seed".to_string(), Value::from(train_random_seed));

    train_map.insert("train_cfg".to_string(), train_config);

    // reg config
    let mut reg_config = HashMap::new();
    reg_config.insert("name".to_string(), Value::from(reg_name));
    reg_config.insert("alpha".to_string(), Value::from(reg_alpha));

    train_map.insert("reg_cfg".to_string(), reg_config);

    // opt config
    let mut opt_config = HashMap::new();
    opt_config.insert("name".to_string(), Value::from(opt_name));
    opt_config.insert("lr".to_string(), Value::from(opt_lr));
    opt_config.insert("momentum".to_string(), Value::from(opt_momentum));
    opt_config.insert("weight_decay".to_string(), Value::from(opt_weight_decay));
    opt_config.insert("precision".to_string(), Value::from(opt_precision));

    train_map.insert("opt_cfg".to_string(), opt_config);

    // SELECT train('mlp', 784, 10, 100, true, 'mnist', '/tmp/mnist', 2, 16, 'cpu', 'float32', 0, 'L2', '0.5', 'sgd', '0.01', '0.9', '0.0001', 'float32');
    let args_json = json!(train_map).to_string();
    crate::bindings::trainer::train(&args_json).to_string()
}


#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "train_result")]
fn train_result(task_id: String) -> String {
    let mut args_map = HashMap::new();
    args_map.insert("task_id".to_string(), Value::from(task_id));
    let args_json = json!(args_map).to_string();
    crate::bindings::trainer::train_result(&args_json).to_string()
}








