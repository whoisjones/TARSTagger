import argparse
from argparse import Namespace

from prettytable.prettytable import PrettyTable

import yaml

def read_config(path):
    """Return namespace object like argparser of yaml file"""

    with open(path, "r") as f:
        conf = yaml.safe_load(f)

    table = PrettyTable(["Parameter", "Value"])

    for parameter, value in conf.items():
        table.add_row([parameter, value])

    print("Configrurations:")
    print(table)
    return Namespace(**conf)

def argparse_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
                        help="Configration file (YAML) for all arguments, if empty, use command lines arguments")
    parser.add_argument("--cuda", type=bool, default=True, help="Whether to use CUDA or not.")
    parser.add_argument("--cuda_devices", type=str, default="0", help="If multiple devices are present, select which ones to train on.")
    parser.add_argument("--output_dir", type=str, help="Where to store the model")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["baseline", "baseline_kshot", "tars", "tars_kshot"],
        help="What experiment should be performed.",
    )

    parser.add_argument("--runs", type=int, default=1, help="How many runs should be performed.")
    parser.add_argument("--set_seed", type=bool, default=False, help="Whether to set seed, i.e. for k-shot sampling.")
    parser.add_argument("--final_test", type=bool, default=True, help="Whether to perform final test on data.")

    parser.add_argument(
        "--language_model",
        type=str,
        default="xlm-roberta-large",
        help="Pretrained language model name from huggingface hub or local path.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="conll",
        help="Corpus to train on, select one of the supported in corpora package.",
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        default=False,
        help="Whether to store the model on disk.",
    )
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--reuse_decoder_weights", type=bool, default=False)
    parser.add_argument("--reuse_corpus_for_weights", type=str, default="conll")
    return parser.parse_args()
