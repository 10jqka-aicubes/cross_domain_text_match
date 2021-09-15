#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath

echo 'GREETINGS FROM THS! NOW START TESTING...'
pwd
# -e参数指定训练轮数，使用-b参数指定batch size
# -m参数中指定所使用的bert模型路径，支持Huggingface/transformers model (如 BERT, RoBERTa, XLNet, XLM-R)
# -i指定训练文件地址 -o指定模型保存地址
python train.py -e 1 -b 16 -i $TRAIN_FILE_DIR/train.tsv -o $SAVE_MODEL_DIR -m "/read-only/common/pretrain_model/transformers/bert-base-chinese/"
echo 'TRAIN FINISHED'