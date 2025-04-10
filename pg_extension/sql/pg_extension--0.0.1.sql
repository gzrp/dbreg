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


/*
This file is auto generated by pgrx.

The ordering of items is not stable, it is driven by a dependency graph.
*/
/* </end connected objects> */

/* <begin connected objects> */
-- src/lib.rs:38
-- pg_extension::echo_python
CREATE  FUNCTION "echo_python"(
	"message" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'echo_python_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/lib.rs:31
-- pg_extension::hello_pg_extension
CREATE  FUNCTION "hello_pg_extension"() RETURNS TEXT /* &str */
STRICT
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'hello_pg_extension_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/lib.rs:47
-- pg_extension::train
CREATE  FUNCTION "train"(
	"model_name" TEXT, /* alloc::string::String */
	"model_in_features" INT, /* i32 */
	"model_out_features" INT, /* i32 */
	"model_hidden_features" INT, /* i32 */
	"model_bias" bool, /* bool */
	"data_name" TEXT, /* alloc::string::String */
	"data_dir_path" TEXT, /* alloc::string::String */
	"train_max_epoch" INT, /* i32 */
	"train_batch_size" INT, /* i32 */
	"train_device" TEXT, /* alloc::string::String */
	"train_precision" TEXT, /* alloc::string::String */
	"train_random_seed" INT, /* i32 */
	"reg_name" TEXT, /* alloc::string::String */
	"reg_alpha" TEXT, /* alloc::string::String */
	"opt_name" TEXT, /* alloc::string::String */
	"opt_lr" TEXT, /* alloc::string::String */
	"opt_momentum" TEXT, /* alloc::string::String */
	"opt_weight_decay" TEXT, /* alloc::string::String */
	"opt_precision" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'train_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/lib.rs:120
-- pg_extension::train_result
CREATE  FUNCTION "train_result"(
	"task_id" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'train_result_wrapper';
/* </end connected objects> */
