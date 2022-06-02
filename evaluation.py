import os
import re
import json
import numpy as np

def evaluate():
    directories = list(filter(lambda x: "shot" in x, os.listdir("resources")))
    directories = sorted(directories, key=lambda x: (x.split("_")[0], x.split("_")[1], int(re.findall(r'\d+', x.split("_")[-1]).pop())))

    for directory in directories:
        print(50*'-')
        if "baseline" in directory:
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
            print(f"{directory}")
            print(f"micro avg: {round(np.average(micro_f1), 3) * 100}")
            print(f"micro avg: {round(np.std(micro_f1), 3) * 100}")
            print(f"macro avg: {round(np.average(macro_f1), 3) * 100}")
            print(f"micro avg: {round(np.std(macro_f1), 3) * 100}")
            print(f"weighted : {round(np.average(weighted), 3) * 100}")
            print(f"micro avg: {round(np.std(weighted), 3) * 100}")
        elif "tars" in directory:
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
            print(f"{directory}")
            print(f"micro avg: {round(np.average(micro_f1), 3) * 100}")
            print(f"macro avg: {round(np.average(macro_f1), 3) * 100}")
            print(f"weighted : {round(np.average(weighted), 3) * 100}")
        else:
            raise ValueError(f"neither tars nor baseline included in folder name: {directory}")

if __name__ == "__main__":
    evaluate()