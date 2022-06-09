import argparse
from src.models import baseline


def run(args):
    if args.experiment == "baseline":
        baseline(args)


if __name__ == "__main__":
    args = argparse_train()
    run(args)
