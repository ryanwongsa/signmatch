import pandas as pd


def class_list(dictionary, spottings):
    spotted = dictionary["id"].isin(set(spottings["label"]))
    return dictionary[spotted].reset_index(drop=True)


def class_index(classes):
    return {gloss: index for index, gloss in enumerate(classes["id"])}


def num_classes(manifest):
    return len(pd.read_csv(manifest))
