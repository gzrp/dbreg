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

use log::error;
use pyo3::prelude::*;
use once_cell::sync::Lazy;
use pyo3::ffi::c_str;
use pyo3::IntoPyObjectExt;
use pyo3::types::PyTuple;

pub fn run_python_train_function(
    py_module: &Lazy<Py<PyModule>>,
    parameters: &String,
    function_name: &str,
) -> serde_json::Value {
    let parameters_str = parameters.to_string();
    let results = Python::with_gil(|py| -> String {
        let run_script: Py<PyAny> = py_module.getattr(py, function_name).unwrap().into();


        let result = run_script.call1(py, PyTuple::new(py, &[parameters_str.into_pyobject(py)]))?;

        let result = match result {
            Err(e) => {
                let traceback = e.traceback(py).unwrap().format().unwrap();
                error!("{traceback} {e}");
                format!("{traceback} {e}")
            }
            Ok(o) => o.extract(py).unwrap(),
        };
        result
    });
    serde_json::from_str(&results).unwrap()
}



// Python Train Module  CARGO_MANIFEST_DIR cargo.toml dir
pub static PY_TRAIN_MODULE: Lazy<Py<PyModule>> = Lazy::new(|| {
    Python::with_gil(|py| -> Py<PyModule> {
        let src = c_str!(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../src/pg_interface.py")));
        PyModule::from_code(py, src, c_str!(""), c_str!("")).unwrap().into()
    })
});