#!/bin/bash

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

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <host> <port> <user> <dataset_path> <dataset_name>"
  exit 1
fi

HOST="$1"
PORT="$2"
USER_NAME="$3"
DATASET_PATH="$4"
DATASET_NAME="$5"


DB_NAME="pg_extension"

types=("train" "valid" "test")

echo $HOST
echo $PORT
echo $DATASET_PATH
echo $DATASET_NAME


echo "step1: remove old csv file if exists..."
for type in "${types[@]}"; do
  if [ -f "${DATASET_PATH}/${type}.csv" ]; then
    echo "remove old ${DATASET_PATH}/${type}.csv"
    rm "${DATASET_PATH}/${type}.csv"
  fi
done

echo "step2: create new csv file from libsvm file ..."
for type in "${types[@]}"; do
  echo "create new ${DATASET_PATH}/${type}.csv"
  awk '{
        for (i = 1; i <= NF; i++) {
            printf "%s", $i;  # print each field as-is
            if (i < NF) {
                printf " ";  # if its not the last field, print a space
            }
        }
        if (NR < FNR) {
          printf "\n";  # end of line
        }
    }' "${DATASET_PATH}/${type}.libsvm" > "${DATASET_PATH}/${type}.csv"
done

# sudo ./load_libsvm_dataset_to_db.sh 127.0.0.1 28814 zrp /tmp/pycharm_project_dbreg/resources/dataset/frappe frappe
#
echo "step3: create database pg_extension ..."
createdb -h $HOST -p $PORT -U $USER_NAME $DB_NAME

echo "step4: create the table for database ..."
for type in "${types[@]}"; do
  echo "drop old  ${DATASET_NAME}_${type} table"
  drop_table_cmd="DROP TABLE ${DATASET_NAME}_${type}"
  echo $drop_table_cmd | psql -h $HOST -p $PORT -U $USER_NAME -d $DB_NAME

  num_columns=$(awk 'NF > max { max = NF } END { print max }' "${DATASET_PATH}/${type}.libsvm")
  create_table_cmd="CREATE TABLE IF NOT EXISTS ${DATASET_NAME}_${type} (id SERIAL PRIMARY KEY, label INTEGER"

  for (( i=2; i<=$num_columns; i++ )); do
    create_table_cmd+=", col$(($i-1)) TEXT"
  done
  create_table_cmd+=");"

  echo "create ${DATASET_NAME}_${type} table ..."
  echo $create_table_cmd
  echo $create_table_cmd | psql -h $HOST -p $PORT -U $USER_NAME -d $DB_NAME
done

echo "step5: import data to postgres table ..."
for type in "${types[@]}"; do
  num_columns=$(awk 'NF > max { max = NF } END { print max }' "${DATASET_PATH}/${type}.libsvm")
  columns="label"
  for (( i=2; i<=num_columns; i++ )); do
    columns+=", col$(($i-1))"
  done
  echo "loading ${type}.csv into postgres"
  echo ${columns}
  psql -h $HOST -p $PORT -U $USER_NAME -d $DB_NAME -c "\COPY  ${DATASET_NAME}_${type}($columns) FROM '${DATASET_PATH}/${type}.csv' DELIMITER ' '"
done

echo "data load finish."









