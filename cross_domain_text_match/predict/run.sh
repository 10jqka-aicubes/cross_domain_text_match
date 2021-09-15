#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath

mkdir -p $PREDICT_RESULT_FILE_DIR
python predict.py -m $SAVE_MODEL_DIR -i $PREDICT_FILE_DIR/test.tsv -o $PREDICT_RESULT_FILE_DIR
echo 'PREDICT FINISHED 预测结果保存在输出目录下的predict.json文件'