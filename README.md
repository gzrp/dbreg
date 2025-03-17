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


# dbreg

<img alt="img.png" src="img.png" width="500"/>


```postgresql
-- train_base(mdict: Json, odict:Json, ddict:Json, tdict:Json, vdict:Json) -> String
-- mdict = {"name": "mlp", "in_features":10, "out_features":2, "hidden_features":16, "bias": true}
-- odict = {"name": "sgd", "lr":0.01, "momentum":0.9, "weight_decay":0.00001, "precision": "float32"}
-- ddict = {"type":"stream", "svc_url": "http://192.168.56.20:8094", "table_name": "frappe_train", "namespace": "train", "columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"], "batch_size": 16}
-- vdict = {"type":"stream", "svc_url": "http://192.168.56.20:8094", "table_name": "frappe_test", "namespace": "test", "columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"], "batch_size": 16}
-- tdict = {"device": "cpu", "seed":0, "max_epoch":3}
select train_base('{"name": "mlp", "in_features":10, "out_features":2, "hidden_features":16, "bias": true}','{"name": "sgd", "lr":0.01, "momentum":0.9, "weight_decay":0.00001, "precision": "float32"}','{"type":"stream", "svc_url": "http://192.168.56.20:8094", "table_name": "frappe_train", "namespace": "train", "columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"], "batch_size": 16}','{"device": "cpu", "seed":0, "max_epoch":3}','{"type":"stream", "svc_url": "http://192.168.56.20:8094", "table_name": "frappe_test", "namespace": "test", "columns": ["label","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10"], "batch_size": 16}')
```

