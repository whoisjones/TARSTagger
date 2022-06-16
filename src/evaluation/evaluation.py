import os
import re
import json
import numpy as np
import pandas as pd

def evaluate(directory):
    micro_f1 = []
    macro_f1 = []
    weighted = []
    for run in os.listdir(directory):
        with open(f"{directory}/{run}/results.txt") as f:
            for line in f.readlines():
                if "micro avg" in line:
                    micro_f1.append(float(line.split()[-2]))
                if "macro avg" in line:
                    macro_f1.append(float(line.split()[-2]))
                if "weighted avg" in line:
                    weighted.append(float(line.split()[-2]))
    print(f"micro avg: {round(np.average(micro_f1) * 100, 2)} \n"
          f"micro std: {round(np.std(micro_f1) * 100, 2)} \n"
          f"macro avg: {round(np.average(macro_f1) * 100, 2)} \n"
          f"macro std: {round(np.std(macro_f1) * 100, 2)} \n"
          f"weigh avg: {round(np.average(weighted) * 100, 2)} \n"
          f"weigh std: {round(np.std(weighted) * 100, 2)} \n"
          )



def evaluate_to_latex():
    directories = list(filter(lambda x: "shot" in x, os.listdir("resources")))
    directories = sorted(directories, key=lambda x: (x.split("_")[0], x.split("_")[1], int(re.findall(r'\d+', x.split("_")[-1]).pop())))

    kshots = ["0shot", "1shot", "2shot", "4shot", "8shot", "16shot", "32shot", "64shot"]

    output_map = {}
    for directory in directories:
        name = f"{directory.split('_')[0]} {directory.split('_')[1]}"
        if not name in output_map:
            output_map[name] = {kshot: {} for kshot in kshots}

    df = pd.DataFrame()
    df["k"] = kshots

    for directory in directories:
        name = f"{directory.split('_')[0]} {directory.split('_')[1]}"
        k_shot = directory.split("_")[-1]
        if "baseline" in directory and not "0shot" in directory:
            micro_f1 = []
            macro_f1 = []
            weighted = []
            for run in os.listdir(f"resources/{directory}"):
                with open(f"resources/{directory}/{run}/all_results.json") as f:
                    data = json.load(f)
                    results = data["predict_classification_report"]
                    for line in results.splitlines():
                        if "micro avg" in line:
                            micro_f1.append(float(line.split()[-2]))
                        if "macro avg" in line:
                            macro_f1.append(float(line.split()[-2]))
                        if "weighted avg" in line:
                            weighted.append(float(line.split()[-2]))
        elif "tars" in directory and not "0shot" in directory:
            micro_f1 = []
            macro_f1 = []
            weighted = []
            for run in os.listdir(f"resources/{directory}"):
                with open(f"resources/{directory}/{run}/results.txt") as f:
                    for line in f.readlines():
                        if "micro avg" in line:
                            micro_f1.append(float(line.split()[-2]))
                        if "macro avg" in line:
                            macro_f1.append(float(line.split()[-2]))
                        if "weighted avg" in line:
                            weighted.append(float(line.split()[-2]))
        elif "0shot" in directory:
            micro_f1 = []
            macro_f1 = []
            weighted = []
            with open(f"resources/{directory}/results.txt") as f:
                for line in f.readlines():
                    if "micro avg" in line:
                        micro_f1.append(float(line.split()[-2]))
                    if "macro avg" in line:
                        macro_f1.append(float(line.split()[-2]))
                    if "weighted avg" in line:
                        weighted.append(float(line.split()[-2]))
        else:
            raise ValueError(f"neither tars nor baseline included in folder name: {directory}")

        output_map[name][k_shot] = {
            "micro avg": round(np.average(micro_f1) * 100, 2),
            "micro std": round(np.std(micro_f1) * 100, 2),
            "macro avg": round(np.average(macro_f1) * 100, 2),
            "macro std": round(np.std(macro_f1) * 100, 2),
            "weighted avg": round(np.average(weighted) * 100, 2),
            "weighted std": round(np.std(weighted) * 100, 2),
        }

    for name, results in output_map.items():
        df[name] = [f"{results[kshot]['weighted avg']} pm {results[kshot]['weighted std']}" for kshot in kshots]

    print(df.to_latex(index=False))