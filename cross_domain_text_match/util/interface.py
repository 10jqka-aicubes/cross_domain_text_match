#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Filename:
#
# Description:
# Version:       1.0
# Company:       www.10jqka.com.cn
#
# -------------------------------------------#
from abc import ABC, abstractmethod
from pathlib import Path


class TrainInterface(ABC):
    def __init__(self, input_file_dir: Path, save_model_dir: Path, *args, **kargs):
        self.initialize(input_file_dir, save_model_dir, *args, **kargs)
        self.input_file_dir = input_file_dir
        self.save_model_dir = save_model_dir

    @abstractmethod
    def initialize(self, input_file_dir: Path, save_model_dir: Path, *args, **kargs):
        """训练初始化函数
        Args:
            input_file_dir (Path): 训练文件的目录
            save_model_dir (Path): 保存模型的目录
        """
        raise NotImplementedError()

    @abstractmethod
    def do_train(self):
        raise NotImplementedError()


class PredictInterface(ABC):
    def __init__(self, input_file_dir: Path, load_model_dir: Path, predict_file_dir: Path, *args, **kargs):
        self.initialize(input_file_dir, load_model_dir, predict_file_dir, *args, **kargs)
        self.input_file_dir = input_file_dir
        self.load_model_dir = load_model_dir
        self.predict_file_dir = predict_file_dir

    @abstractmethod
    def initialize(self, input_file_dir: Path, load_model_dir: Path, predict_file_dir: Path, *args, **kargs):
        """预测初始化函数
        Args:
            input_file_dir (Path): 预测的文件目录
            load_model_dir (Path): 加载模型的目录
            predict_file_dir (Path): 预测结果的文件目录
        """
        raise NotImplementedError()

    @abstractmethod
    def do_predict(self):
        raise NotImplementedError()


class MetricsInterface(ABC):
    @abstractmethod
    def do_eval(
        self,
        predict_file_dir: Path,
        groudtruth_file_dir: Path,
        result_json_file: Path,
        result_detail_file: Path,
        *args,
        **kargs
    ):
        """评测主函数

        Args:
            predict_file_dir (Path): 模型预测结果的文件目录
            groudtruth_file_dir (Path): 真实结果的文件目录
            result_json_file (Path): 评测结果，json格式，{"f1": 0.99}
            result_detail_file (Path): 预测明细，可选
        """
        raise NotImplementedError()
