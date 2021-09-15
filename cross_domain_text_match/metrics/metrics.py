# -*- coding:utf-7 -*-
import os
import argparse
import logging
import random
import json
from sklearn.metrics import classification_report, f1_score


class Evaluator:
    def __init__(self, input_path, output_directory, true_label_path):
        """
        initiation works
        """
        self.input_path = input_path
        self.output_directory = output_directory
        self.true_label_path = true_label_path

    def evaluate(self):
        """
        TO DO:
        read data in input file path, write evaluation result to output file
        :param input_path, string format
        :return output_file, string format
        """
        output_file = "result.json"
        result = self.load_data()
        output_path = self.evaluate_metrics(
            test_y=result["label_value"],
            pred_label=result["pred_label"],
            # download_path=os.path.join(self.output_directory, output_file),
            download_path=self.output_directory
        )
        return output_path

    def load_data(self):
        """
        DO NOT MODIFY
        evaluation code seg
        eval mode:
            if true_label_path is provided, return true evaluation result
        test mode:
            if true_label_path is null, randomly generating from {0,1} as true label
            test mode is for testing result structure purpose only
            input path will be replaced by our test data with true label value during evaluation
        """
        input_path = self.input_path
        true_label_path = self.true_label_path
        with open(input_path, "r") as fp:
            preds = fp.read().splitlines()
            preds = [json.loads(p) for p in preds]
        result = dict()
        result["pred_label"] = [p["label"] for p in preds]

        if true_label_path is None:
            # test mode
            result["label_value"] = [
                1 if random.random() >= 0.5 else 0
                for _ in range(len(result["pred_label"]))
            ]
        else:
            # eval mode
            with open(true_label_path, "r") as fp:
                true_label = fp.read().splitlines()
                true_label = [json.loads(p) for p in true_label]
            result["label_value"] = [p["label"] for p in true_label]
        return result

    def evaluate_metrics(self, test_y, pred_label, download_path=None):
        if download_path is not None:
            result = dict()
            result["F1"] = f1_score(test_y, pred_label, average="binary")
            result = json.dumps(result)
            with open(download_path, "w") as f:
                f.write(result)
        print(
            f"--classification report --+AFw-n{classification_report(test_y,  pred_label)}"
        )
        return download_path


def main():
    parser = argparse.ArgumentParser()

    # FORMAT: python evaluate.py -i <input_path> -o <output_directory>
    parser.add_argument("-i", "--input_path", dest="input_path", help="input file path")
    parser.add_argument(
        "-o",
        "--output_directory",
        dest="output_directory",
        help="output file directory",
    )
    parser.add_argument(
        "-t" "--true_label",
        dest="true_label_path",
        help="true label file path",
        default=None,
    )
    parser.add_argument(
        "-l", "--log", dest="loglevel", help="output file directory", default="warning"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    evaluator = Evaluator(
        input_path=args.input_path,
        output_directory=args.output_directory,
        true_label_path=args.true_label_path,
    )
    output = evaluator.evaluate()
    logging.info(f"evaluation output saved to: {output}")


if __name__ == "__main__":
    main()
