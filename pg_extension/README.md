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

select train('{"name": "mlp", "in_features":10, "out_features":2, "hidden_features":100, "bias": true, "reg": {"name":"L2", "alpha": 1.0}}','{"name": "sgd", "lr":0.01, "momentum":0.9, "weight_decay":0.00001}','{"device": "cpu", "seed":0, "max_epoch":10}','{"table_name": "frappe_train", "namespace": "train", "batch_size": 64,"columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"]}','{"table_name": "frappe_test", "namespace": "test", "batch_size": 64,"columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"]'});
```

run cmd

```shell
cd ./pg_extension
cargo pgrx run
SELECT * from pg_available_extensions;
DROP EXTENSION IF EXISTS pg_extension;
CREATE EXTENSION pg_extension;
\dx+ pg_extension;

SELECT hello_pg_extension('{}');
SELECT train('{"name": "mlp", "in_features":10, "out_features":2, "hidden_features":100, "bias": true, "reg": {"name":"L2", "alpha": 1.0}}','{"name": "sgd", "lr":0.01, "momentum":0.9, "weight_decay":0.00001}','{"device": "cpu", "seed":0, "max_epoch":3}','{"table_name": "frappe_train", "namespace": "train", "batch_size": 64,"columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"]}','{"table_name": "frappe_test", "namespace": "test", "batch_size": 64,"columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"]}');

```

```
fn train(mdict: Json, odict:Json, tdict:Json, ddict:Json, vdict:Json) -> String
select train(
	'{
		"name": "mlp", 
		"in_features":10, "out_features":2, "hidden_features":100, "bias": true, 
		"reg": {"name":"L2", "alpha": 1.0}
	}',
	'{
		"name": "sgd", 
		"lr":0.01, "momentum":0.9, "weight_decay":0.00001
	}',
	'{
		"device": "cpu", "seed":0, 
		"max_epoch":10
	}',
	'{
		"table_name": "frappe_train", "namespace": "train", "batch_size": 64,
		"columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"] 
	}',
	'{
		"table_name": "frappe_test", "namespace": "test", "batch_size": 64,
		"columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"]
	}',
);
```


```json
{
    "code": 200,
    "task_id": "1748c724055e11f0be9169914616b394",
    "train_total_time": "201.25s",
    "result": {
        "records": [
            {
                "epoch": 0,
                "loss": "1903.75",
                "train_acc": "66.53%",
                "test_acc": "66.95%",
                "train_time": "61.01s",
                "val_time": "5.46s"
            },
            {
                "epoch": 1,
                "loss": "1867.47",
                "train_acc": "66.54%",
                "test_acc": "66.95%",
                "train_time": "57.82s",
                "val_time": "5.47s"
            },
            {
                "epoch": 2,
                "loss": "1861.89",
                "train_acc": "66.54%",
                "test_acc": "66.95%",
                "train_time": "63.38s",
                "val_time": "5.74s"
            }
        ]
    },
    "time_usage": "201.53s"
}
```