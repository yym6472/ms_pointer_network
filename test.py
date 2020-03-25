import argparse
import math
from typing import Any, Union, Dict, Iterable, List, Optional, Tuple

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from model import MSPointerNetwork
from dataset_reader import MSDatasetReader
from predictor import MSPointerPredictor
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer

from allennlp.data import vocabulary


def main(args):
    archive = load_archive(args.output_dir)
    predictor = Predictor.from_archive(archive=archive, predictor_name="ms_pointer")

    source1 = "任天堂 switch 主机 全新 一代 游戏机 体感 家用 电视"
    source2 = "任天堂 Nintendo 游戏机"
    output_dict = predictor.predict(source1, source2)
    log_probs = output_dict["predicted_log_probs"]
    predicted_tokens = output_dict["predicted_tokens"]
    
    print(f"Top {len(log_probs)} predictions:")
    for log_prob, tokens in zip(log_probs, predicted_tokens):
        prob = math.exp(log_prob)
        sentence = " ".join(tokens)
        print(f"    {prob:.2f}    {sentence}")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--output_dir", type=str, default="./output/ms_pointer/",
                            help="the directory that stores training output")
    args = arg_parser.parse_args()
    main(args)