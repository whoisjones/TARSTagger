import os
import re
import json
import numpy as np
import pandas as pd


def total_eval(directory):
    for group in ["A", "B", "C"]:
        all_shots = list(filter(lambda x: x.startswith(group), os.listdir(directory)))
        one_shots = list(filter(lambda k: k.endswith("1shot"), all_shots))
        five_shots = list(filter(lambda k: k.endswith("5shot"), all_shots))
        mif1 = []
        mif1sd = []
        maf1 = []
        maf1sd = []
        wgf1 = []
        wgf1sd = []
        for exp in one_shots:
            mif1avg, mif1std, maf1avg, maf1std, wgtf1avg, wgtf1std = evaluate(f"{directory}/{exp}")
            mif1.append(mif1avg)
            mif1sd.append(mif1std)
            maf1.append(maf1avg)
            maf1sd.append(maf1std)
            wgf1.append(wgtf1avg)
            wgf1sd.append(wgtf1std)

        print(f"Group {group}")
        print("1-Shot")
        print(f"Average micro f1: {np.average(mif1)}")
        print(f"std avg micro f1: {np.average(mif1sd)}")
        print(f"Average macro f1: {np.average(maf1)}")
        print(f"std avg macro f1: {np.average(maf1sd)}")
        print(f"Average micro f1: {np.average(wgf1)}")
        print(f"std avg micro f1: {np.average(wgf1sd)}")
        print(f"max (micro): {np.max(mif1)} with {mif1sd[np.argmax(mif1)]}")
        print(f"max (macro): {maf1[np.argmax(mif1)]} with {maf1sd[np.argmax(mif1)]}")
        print(f"max (weigh): {wgf1[np.argmax(mif1)]} with {wgf1sd[np.argmax(mif1)]}")

        mif1 = []
        mif1sd = []
        maf1 = []
        maf1sd = []
        wgf1 = []
        wgf1sd = []
        for exp in five_shots:
            mif1avg, mif1std, maf1avg, maf1std, wgtf1avg, wgtf1std = evaluate(f"{directory}/{exp}")
            mif1.append(mif1avg)
            mif1sd.append(mif1std)
            maf1.append(maf1avg)
            maf1sd.append(maf1std)
            wgf1.append(wgtf1avg)
            wgf1sd.append(wgtf1std)

        print(f"Group {group}")
        print("5-Shot")
        print(f"Average micro f1: {np.average(mif1)}")
        print(f"std avg micro f1: {np.average(mif1sd)}")
        print(f"Average macro f1: {np.average(maf1)}")
        print(f"std avg macro f1: {np.average(maf1sd)}")
        print(f"Average micro f1: {np.average(wgf1)}")
        print(f"std avg micro f1: {np.average(wgf1sd)}")
        print(f"max (micro): {np.max(mif1)} with {mif1sd[np.argmax(mif1)]}")
        print(f"max (macro): {maf1[np.argmax(mif1)]} with {maf1sd[np.argmax(mif1)]}")
        print(f"max (weigh): {wgf1[np.argmax(mif1)]} with {wgf1sd[np.argmax(mif1)]}")

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
    return (round(np.average(micro_f1) * 100, 2), round(np.std(micro_f1) * 100, 2),
            round(np.average(macro_f1) * 100, 2), round(np.std(macro_f1) * 100, 2),
            round(np.average(weighted) * 100, 2), round(np.std(weighted) * 100, 2))



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