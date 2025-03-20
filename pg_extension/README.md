<!--
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with < this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
-->

# cargo pgrx

```shell
# 1. create a new pg extension 
cargo pgrx new pg_extension
# 2. init pg env
cargo pgrx init
# 3. cd pg_extension
cargo pgrx run
# 4. DROP EXTENSION IF EXISTS pg_extension;
pg_extension=# CREATE EXTENSION pg_extension;
CREATE EXTENSION
pg_extension=# \dx+ pg_extension;
pg_extension=# SELECT hello_pg_extension('{}');
hello_pg_extension

select * from pg_available_extensions;

select train_base('{"name": "mlp", "in_features":10, "out_features":2, "hidden_features":100, "bias": true, "reg": {"name":"L2", "alpha": 1.0}}','{"name": "sgd", "lr":0.01, "momentum":0.9, "weight_decay":0.00001, "precision": "float32"}','{"type":"stream", "svc_url": "http://192.168.56.20:8094", "table_name": "frappe_train", "namespace": "train", "columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"], "batch_size": 64}','{"device": "cpu", "seed":0, "max_epoch":10}','{"type":"stream", "svc_url": "http://192.168.56.20:8094", "table_name": "frappe_test", "namespace": "test", "columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"], "batch_size": 64}');

```
