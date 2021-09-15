# 【9-10双月赛】跨领域迁移的文本语义匹配

​	FAQ ( Frequently Asked Questions )问答系统，一般根据用户的query从问答数据集中检索出相似的问题，然后返回对应的答案，文本语义匹配是其中的关键。而构建有效的文本语义匹配模型需要依赖大量的标注数据。然而，实际业务中不同领域的数据分布具有差异，为每个领域分别构建训练数据是成本高昂的（时间、人力等）。因此需要利用迁移学习、领域适应等技术解决跨领域文本语义匹配问题。该问题的难点在于目标领域通常只有少量的语料，而大量的现有数据和目标领域数据之间可能存在显著的分布差异。选手需要利用迁移学习等技术，借助现有的语料提升目标任务的效果。

- 本代码是该赛题的一个基础demo，仅供参考学习。


- 比赛地址：http://contest.aicubes.cn/	


- 时间：2021-09 ~ 2021-10



## 如何运行Demo

- clone代码


- 准备预训练模型

  - 下载模型 [bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main)
  - 所有模型文件统一下载到一个目录下

- 准备环境

  - cuda10.0以上
  - python3.7以上
  - 安装python依赖

  ```
  python -m pip install -r requirements.txt
  ```

- 准备数据，从[官网](http://contest.aicubes.cn/#/detail?topicId=23)下载数据

  - 训练数据`train.tsv`，放在训练数据目录中
  - 预测数据`test.tsv` ，放在预测目录下

- 调整参数配置，参考[模板项目](https://github.com/10jqka-aicubes/project-demo)的说明

  - `cross_domain_text_match/setting.conf`
  - 其他注意下`run.sh`里使用的参数，比如指定预训练模型的路径

- 运行

  - 训练

  ```
  bash cross_domain_text_match/train/run.sh
  ```

  - 预测

  ```
  bash cross_domain_text_match/predict/run.sh
  ```

  - 计算结果指标

  ```
  bash cross_domain_text_match/metrics/run.sh
  ```



## 反作弊声明

1）参与者不允许在比赛中抄袭他人作品、使用多个小号，经发现将取消成绩；

2）参与者禁止在指定考核技术能力的范围外利用规则漏洞或技术漏洞等途径提高成绩排名，经发现将取消成绩；

3）在A榜中，若主办方认为排行榜成绩异常，需要参赛队伍配合给出可复现的代码。



## 赛事交流

![同花顺比赛小助手](http://speech.10jqka.com.cn/arthmetic_operation/245984a4c8b34111a79a5151d5cd6024/客服微信.JPEG)