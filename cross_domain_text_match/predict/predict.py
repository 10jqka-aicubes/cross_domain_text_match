# -*- coding:utf-8 -*-
import logging
import os
import argparse
import json
from sentence_transformers import models
from sentence_transformers import SentenceTransformer, InputExample
from sklearn.metrics.pairwise import paired_cosine_distances


logger = logging.getLogger(__name__)


def load_data(file):
    with open(file, "r") as fp:
        lines = fp.read().splitlines()
        lines = [line.strip().split("\t") for line in lines]
    data = []
    if len(lines[0]) == 2:
        for row in lines:
            data.append(InputExample(texts=[row[0], row[1]], label=0))
    else:
        for row in lines:
            data.append(InputExample(texts=[row[1], row[2]], label=int(row[0])))
    return data


class Predictor:
    def __init__(self, input_path, output_directory):
        """
        initiation works
        """
        self.input_path = input_path
        self.output_directory = output_directory

    def predict(self, args):
        """
        TO DO:
        read data in input file path, write prediction result to output file
        :param input_path, string format
        :return output_file, string format
        """
        # 输出预测结果到文件
        output_file = "predict.json"
        output_file = os.path.join(self.output_directory, output_file)
        # 从文件中读取训练数据
        test_data = load_data(self.input_path)

        # 加载模型
        model_name = args.model_path
        word_embedding_model = models.Transformer(model_name)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        sentences1 = [example.texts[0] for example in test_data]
        sentences2 = [example.texts[1] for example in test_data]

        sentences = list(set(sentences1 + sentences2))
        embeddings = model.encode(
            sentences, batch_size=2, show_progress_bar=True, convert_to_numpy=True
        )
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in sentences1]
        embeddings2 = [emb_dict[sent] for sent in sentences2]
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        preds = [1 if s >= 0.91 else 0 for s in cosine_scores]
        preds = [json.dumps({"label": prob}) for prob in preds]
        # 保存预测结果
        with open(output_file, "w") as fp:
            fp.write("\n".join(preds))
        return output_file


def main():
    parser = argparse.ArgumentParser()

    # FORMAT: python predict.py -i <input_path> -o <output_directory>
    parser.add_argument(
        "-m", "--model_path", dest="model_path", help="input model path"
    )
    parser.add_argument("-i", "--input_path", dest="input_path", help="input file path")
    parser.add_argument(
        "-o",
        "--output_directory",
        dest="output_directory",
        help="output file directory",
    )
    parser.add_argument(
        "-l", "--log", dest="loglevel", help="output file directory", default="warning"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format="%(asctime)s - %(name)s- %(levelname)s - %(message)s",
    )
    predictor = Predictor(
        input_path=args.input_path, output_directory=args.output_directory
    )
    output = predictor.predict(args)
    logging.info(f"prediction output saved to: {output}")


if __name__ == "__main__":
    main()
