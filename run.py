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
        elif args.experiment == "tars_cross_lingual_pretrain":
            tars_cross_lingual_pretrain(args, run)
        elif args.experiment == "tars_cross_lingual_kshot":
            tars_cross_lingual_kshot(args, run)
        elif args.experiment == "tars_cross_lingual_zeroshot":
            tars_cross_lingual_zeroshot(args, run)
        elif args.experiment == "tars_IOscheme_pretrain":
            tars_IOscheme_pretrain(args, run)
        elif args.experiment == "tars_IOscheme_kshot":
            tars_IOscheme_kshot(args, run)
        elif args.experiment == "tars_IOscheme_zeroshot":
            tars_IOscheme_zeroshot(args, run)
        elif args.experiment == "eval_kshot":
            eval_kshot(args, run)
        elif args.experiment  == "eval_crossling_kshot":
            eval_crossling_kshot(args, run)
        else:
            raise ValueError(f"Unknown experiment: {args.experiment}")

if __name__ == "__main__":
    args = argparse_train()

    if args.config != "":
        args = read_config(args.config)

    run_train(args)
