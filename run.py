from src.args import argparse_train, read_config
from src.models import baseline


def run_train(args):
    if args.experiment == "baseline":
        baseline(args)

if __name__ == "__main__":
    args = argparse_train()

    if args.config != "":
        args = read_config(args.config)

    run_train(args)
