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
use pgrx::Json;
::pgrx::pg_module_magic!();

pub mod bindings;


#[pg_extern(name = "hello_pg_extension")]
fn hello_pg_extension(j1: Json) -> String {

    let json_data = serde_json::to_string(&j1).unwrap();
    println!("{}", json_data);
    json_data
//     "Hello, pg_extension111"
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
fn train(mdict: Json, odict:Json, tdict:Json, ddict:Json, vdict:Json) -> String {
    let overall_start_time = Instant::now();
    let args = json!({
        "mdict": mdict,
        "odict": odict,
        "tdict": tdict,
        "ddict": ddict,
        "vdict": vdict
    }).to_string();
    let ans_json = crate::bindings::trainer::train(&args)

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time);
    let overall_elapsed_seconds = overall_elapsed_time.as_secs_f64();

    ans_json.insert("total_time", overall_elapsed_seconds.to_string())
    ans_json.to_string()
}


// #[cfg(feature = "python")]
// #[pg_extern(immutable, parallel_safe, name = "train_result")]
// fn train_result(task_id: String) -> String {
//     let mut args_map = HashMap::new();
//     args_map.insert("task_id".to_string(), Value::from(task_id));
//     let args_json = json!(args_map).to_string();
//     crate::bindings::trainer::train_result(&args_json).to_string()
// }








