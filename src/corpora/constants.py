dataset_key = "dataset_name"
subset_key = "subset"

# SUPPORTED DATASETS
available_datasets = {
    "conll": {dataset_key: "conll2003"},
    "spanish": {dataset_key: "conll2002", subset_key: "es"},
    "dutch": {dataset_key: "conll2002", subset_key: "nl"},
    "finnish": {dataset_key: "finer"},
    "ontonotes": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
    "arabic": {dataset_key: "conll2012_ontonotesv5", subset_key: "arabic_v4"},
    "chinese": {dataset_key: "conll2012_ontonotesv5", subset_key: "chinese_v4"},
    "fewnerd-inter": {dataset_key: "dfki-nlp/few-nerd", subset_key: "inter"},
    "fewnerd-intra": {dataset_key: "dfki-nlp/few-nerd", subset_key: "intra"},
    "fewnerd-supervised": {dataset_key: "dfki-nlp/few-nerd", subset_key: "supervised"},
    "ontonotes_AB": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
    "ontonotes_AC": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
    "ontonotes_BC": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
    "ontonotes_A": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
    "ontonotes_B": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
    "ontonotes_C": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
}

# LABEL MAP FOR TART
tars_label_name_map = {
    "O": "O",
    "B-PER": "person",
    "I-PER": "person",
    "B-ORG": "organization",
    "I-ORG": "organization",
    "B-LOC": "location",
    "I-LOC": "location",
    "B-MISC": "miscellaneous",
    "I-MISC": "miscellaneous",
    "B-PERSON": "person",
    "I-PERSON": "person",
    "B-FAC": "building",
    "I-FAC": "building",
    "B-CARDINAL": "cardinal",
    "I-CARDINAL": "cardinal",
    "B-EVENT": "event",
    "I-EVENT": "event",
    "B-GPE": "geopolitical",
    "I-GPE": "geopolitical",
    "B-LANGUAGE": "language",
    "I-LANGUAGE": "language",
    "B-LAW": "law",
    "I-LAW": "law",
    "B-MONEY": "money",
    "I-MONEY": "money",
    "B-NORP": "affiliation",
    "I-NORP": "affiliation",
    "B-ORDINAL": "ordinal",
    "I-ORDINAL": "ordinal",
    "B-PERCENT": "percentage",
    "I-PERCENT": "percentage",
    "B-PRODUCT": "product",
    "I-PRODUCT": "product",
    "B-QUANTITY": "quantity",
    "I-QUANTITY": "quantity",
    "B-TIME": "time",
    "I-TIME": "time",
    "B-WORK_OF_ART": "art",
    "I-WORK_OF_ART": "art",
    "B-PRO": "product",
    "I-PRO": "product",
    "B-DATE": "date",
    "I-DATE": "date",
}

# TAG SET EXTENSION GROUPS AS IN YANG (2020)
group_a = [
    "B-ORG",
    "B-NORP",
    "B-ORDINAL",
    "B-WORK_OF_ART",
    "B-QUANTITY",
    "B-LAW",
    "I-ORG",
    "I-NORP",
    "I-ORDINAL",
    "I-WORK_OF_ART",
    "I-QUANTITY",
    "I-LAW",
    "O",
]
group_b = [
    "B-GPE",
    "B-CARDINAL",
    "B-PERCENT",
    "B-TIME",
    "B-EVENT",
    "B-LANGUAGE",
    "I-GPE",
    "I-CARDINAL",
    "I-PERCENT",
    "I-TIME",
    "I-EVENT",
    "I-LANGUAGE",
    "O",
]
group_c = [
    "B-PERSON",
    "B-DATE",
    "B-MONEY",
    "B-LOC",
    "B-FAC",
    "B-PRODUCT",
    "I-PERSON",
    "I-DATE",
    "I-MONEY",
    "I-LOC",
    "I-FAC",
    "I-PRODUCT",
    "O",
]

# GROUPING OF DATASETS FOR PROCESSING
standard_datasets = ["conll", "spanish", "dutch", "finnish"]

fewnerd_datasets = ["fewnerd-inter", "fewnerd-intra", "fewnerd-supervised"]

tagset_extension_datasets = [
    "ontonotes_AB",
    "ontonotes_AC",
    "ontonotes_BC",
    "ontonotes_A",
    "ontonotes_B",
    "ontonotes_C",
]

cross_lingual_ontonotes_datasets = ["ontonotes", "arabic", "chinese"]
all_ontonotes = cross_lingual_ontonotes_datasets + tagset_extension_datasets
