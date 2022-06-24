import os
import json
import numpy as np


def total_eval(directory):
    # GROUPS NEED TO BE DEFINED AS IN OUTPUT DIRECTORY
    for group in ["AR", "FI", "ZH"]:
        all_shots = list(filter(lambda x: x.startswith(group), os.listdir(directory)))
        one_shots = list(filter(lambda k: k.endswith("_1shot"), all_shots))
        two_shots = list(filter(lambda k: k.endswith("_2shot"), all_shots))
        four_shots = list(filter(lambda k: k.endswith("_4shot"), all_shots))

        for l, shots in [("1",one_shots), ("2",two_shots), ("4",four_shots)]:
            mif1 = []
            mif1sd = []
            maf1 = []
            maf1sd = []
            wgf1 = []
            wgf1sd = []
            for exp in shots:
                mif1avg, mif1std, maf1avg, maf1std, wgtf1avg, wgtf1std = evaluate(f"{directory}/{exp}")
                mif1.append(mif1avg)
                mif1sd.append(mif1std)
                maf1.append(maf1avg)
                maf1sd.append(maf1std)
                wgf1.append(wgtf1avg)
                wgf1sd.append(wgtf1std)
            print(50*"-")
            print(f"Group {group}")
            print(f"{l}-shot")
            print(f"{len(shots)} runs")
            print(f"Average micro f1: {np.average(mif1)}")
            if l == "0":
                print(f"std avg micro f1: {np.std(mif1)}")
            else:
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
        with open(f'{directory}/{run}/results.txt') as f:
            for line in f.readlines():
                if 'micro avg' in line:
                    micro_f1.append(float(line.split()[-2]))
                if 'macro avg' in line:
                    macro_f1.append(float(line.split()[-2]))
                if 'weighted avg' in line:
                    weighted.append(float(line.split()[-2]))

    return (round(np.average(micro_f1) * 100, 2), round(np.std(micro_f1) * 100, 2),
            round(np.average(macro_f1) * 100, 2), round(np.std(macro_f1) * 100, 2),
            round(np.average(weighted) * 100, 2), round(np.std(weighted) * 100, 2))

def evaluate_json(directory):
    micro_f1 = []
    macro_f1 = []
    weighted = []
    for run in os.listdir(directory):
        with open(f"{directory}/{run}/all_results.json") as f:
            data = json.load(f)
            results = data["predict_classification_report"]
            for line in results.splitlines():
                if "micro avg" in line:
                    micro_f1.append(float(line.split()[-2]))
                if "macro avg" in line:
                    macro_f1.append(float(line.split()[-2]))
                if "weighted avg" in line:
                    weighted.append(float(line.split()[-2]))

    return (round(np.average(micro_f1) * 100, 2), round(np.std(micro_f1) * 100, 2),
            round(np.average(macro_f1) * 100, 2), round(np.std(macro_f1) * 100, 2),
            round(np.average(weighted) * 100, 2), round(np.std(weighted) * 100, 2))
