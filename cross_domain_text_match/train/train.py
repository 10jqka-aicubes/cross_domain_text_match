# -*- coding:utf-8 -*-
"""
train code
"""
import logging
import argparse
import math
from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import BinaryClassificationEvaluator


logger = logging.getLogger(__name__)


def load_data(file):
    with open(file, "r") as fp:
        lines = fp.read().splitlines()
        lines = [line.strip().split("\t") for line in lines]
    data = []
    if len(lines[0]) == 2:
        for row in lines:
            data.append(InputExample(texts=[row[1], row[2]], label=0))
    else:
        for row in lines:
            data.append(InputExample(texts=[row[1], row[2]], label=int(row[0])))
    return data


class Trainer:
    def __init__(
        self, epoch, batch_size, input_path, output_directory, bert_name_or_path
    ):
        """
        initiation works
        """
        self.input_path = input_path
        self.output_directory = output_directory
        self.bert_name_or_path = bert_name_or_path
        self.epoch = int(epoch)
        self.batch_size = int(batch_size)

    def train(self):
        """
        TO DO:
        read data in input file path, train data and dump model files to output directory
        :param input_path, string format
        :return output_path, string format
        """
        # load data from file
        train_samples = load_data(self.input_path)
        # create model
        model_name = self.bert_name_or_path  # pretrained_bert_model
        num_epochs, train_batch_size = self.epoch, self.batch_size
        # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
        word_embedding_model = models.Transformer(model_name)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        train_dataloader = DataLoader(
            train_samples, shuffle=True, batch_size=train_batch_size
        )
        train_loss = losses.SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=2,
        )
        warmup_steps = math.ceil(
            len(train_dataloader) * num_epochs * 0.1
        )  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))
        dev_evaluator = BinaryClassificationEvaluator.from_input_examples(
            train_samples, batch_size=train_batch_size, name="sts-dev"
        )  # 请自行划分验证集
        # 训练模型
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=self.output_directory,
        )
        return self.output_directory


def main():
    parser = argparse.ArgumentParser()

    # FORMAT: python train.py -i <input_path> -o <output_directory>
    parser.add_argument("-e", "--epoch", dest="epoch", help="epoch nums")
    parser.add_argument("-b", "--batch_size", dest="batch_size", help="batch_size")
    parser.add_argument(
        "-i", "--input_path", dest="input_path", help="input file full path"
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        dest="output_directory",
        help="output file full directory",
    )
    parser.add_argument(
        "-l", "--log", dest="loglevel", help="output file directory", default="warning"
    )
    parser.add_argument(
        "-m", "--bert_name_or_path", dest="bert_name_or_path", help="bert_name_or_path"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    trainer = Trainer(
        epoch=args.epoch,
        batch_size=args.batch_size,
        input_path=args.input_path,
        output_directory=args.output_directory,
        bert_name_or_path=args.bert_name_or_path,
    )
    output = trainer.train()
    logging.info(f"train output saved to: {output}")


if __name__ == "__main__":
    main()
