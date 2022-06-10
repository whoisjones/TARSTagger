from src.args import argparse_train, read_config
from src.models import *


def run_train(args):
    for run in range(args.runs):
        if args.experiment == "baseline_pretrain":
            baseline_pretrain(args, run)
        elif args.experiment == "baseline_kshot":
            baseline_kshot(args, run)
        elif args.experiment == "baseline_zeroshot":
            baseline_zeroshot(args, run)
        elif args.experiment == "tars_pretrain":
            tars_pretrain(args, run)
        elif args.experiment == "tars_kshot":
            tars_kshot(args, run)
        elif args.experiment == "tars_zeroshot":
            tars_zeroshot(args, run)
        else:
            raise ValueError(f"Unknown experiment: {args.experiment}")

if __name__ == "__main__":
    args = argparse_train()

    if args.config != "":
        args = read_config(args.config)

    run_train(args)
