#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath


# -i指定预测结果文件，-t指定真实标签文件
# 如果没有指定真实标签文件（-t参数），评估结果为随机值。
python metrics.py -i $PREDICT_RESULT_FILE_DIR/predict.json -o $RESULT_JSON_FILE -t $GROUNDTRUTH_FILE_DIR/test_groundtruth.json
echo 'EVAL FINISHED 结果保存在输出目录下的result.json'