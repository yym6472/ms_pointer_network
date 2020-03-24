import os
import json
import logging
import argparse

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data import DataIterator
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.training import Trainer
from allennlp.training.util import evaluate
from allennlp.common.util import prepare_global_logging, cleanup_global_logging, prepare_environment

from model import MSPointerNetwork
from dataset_reader import MSDatasetReader

from allennlp.data import vocabulary


def main(args):
    params = Params.from_file(args.config_path)
    stdout_handler = prepare_global_logging(args.output_dir, False)
    prepare_environment(params)

    reader = DatasetReader.from_params(params["dataset_reader"])
    train_dataset = reader.read(params.pop("train_data_path", None))
    valid_dataset = reader.read(params.pop("validation_data_path", None))
    test_data_path = params.pop("test_data_path", None)
    if test_data_path:
        test_dataset = reader.read(test_data_path)
        vocab = Vocabulary.from_instances(train_dataset + valid_dataset + test_dataset)
    else:
        test_dataset = None
        vocab = Vocabulary.from_instances(train_dataset + valid_dataset)

    model_params = params.pop("model", None)
    model = Model.from_params(model_params.duplicate(), vocab=vocab)
    vocab.save_to_files(os.path.join(args.output_dir, "vocabulary"))
    # copy config file
    with open(args.config_path, "r", encoding="utf-8") as f_in:
        with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())

    iterator = DataIterator.from_params(params.pop("iterator", None))
    iterator.index_with(vocab)
    
    trainer_params = params.pop("trainer", None)
    trainer = Trainer.from_params(model=model,
                                  serialization_dir=args.output_dir,
                                  iterator=iterator,
                                  train_data=train_dataset,
                                  validation_data=valid_dataset,
                                  params=trainer_params.duplicate())
    trainer.train()

    # evaluate on the test set
    if test_dataset:
        logging.info("Evaluating on the test set")
        import torch  # import here to ensure the republication of the experiment
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "best.th")))
        test_metrics = evaluate(model, test_dataset, iterator,
                                cuda_device=trainer_params.pop("cuda_device", 0),
                                batch_weight_key=None)
        logging.info(f"Metrics on the test set: {test_metrics}")
        with open(os.path.join(args.output_dir, "test_metrics.txt"), "w", encoding="utf-8") as f_out:
            f_out.write(f"Metrics on the test set: {test_metrics}")

    cleanup_global_logging(stdout_handler)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_path", type=str, default="./config/bert.config.json",
                            help="the training config file path")
    arg_parser.add_argument("--output_dir", type=str, default="./output/bert-base/",
                            help="the directory to store output files")
    args = arg_parser.parse_args()
    main(args)