import argparse

from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification

from datasets import load_dataset

import torch
import numpy as np
from seqeval.metrics import classification_report, f1_score

from data import tokenize_and_align_labels


def main():
    # parser training arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xlm-roberta-large")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pretrained_model_path = "resources/baseline/run1"
    model = AutoModelForTokenClassification.from_pretrained(pretrained_model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

if __name__ == "__main__":
    main()
