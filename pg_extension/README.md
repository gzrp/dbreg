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
pg_extension=# \dx+ pg_extension
pg_extension=# SELECT hello_pg_extension();
hello_pg_extension
```
