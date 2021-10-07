#!/bin/bash
set -x

HADOOP_STREAMING_JAR=/opt/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar
IN_DIR=/dataset/AB_NYC_2019.csv
OUT_DIR=/gora_output

hdfs dfs -rm -r $OUT_DIR
chmod a+x mapper.py reducer.py

yarn jar $HADOOP_STREAMING_JAR \
    -files mapper.py,reducer.py \
    -mapper "mapper.py" \
    -reducer "reducer.py" \
    -input $IN_DIR \
    -output $OUT_DIR