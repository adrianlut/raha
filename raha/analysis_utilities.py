from random import sample
from typing import Dict, Tuple

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


def detection_evaluation(d, actual_errors):
    detected_cell_list = list(d.detected_cells.items())

    p_df = DataFrame([list(detection[0]) + list(detection[1]) for detection in detected_cell_list],
                     columns=["row", "column", "p_not", "p"])
    p_df = p_df[["row", "column", "p"]]
    p_df["truth"] = [cell[0] in actual_errors for cell in detected_cell_list]
    p_df["detected"] = True
    #p_df.hist(by="correct", column="p", bins=np.linspace(0.5, 1.0, 21))

    undetected_cell_list = list(d.undetected_cells.items())

    p_df_n = DataFrame([list(detection[0]) + list(detection[1]) for detection in undetected_cell_list],
                       columns=["row", "column", "p_not", "p"])
    p_df_n = p_df_n[["row", "column", "p"]]
    p_df_n["truth"] = [cell[0] in actual_errors for cell in undetected_cell_list]
    p_df_n["detected"] = False

    #p_df_n.hist(by="correct", column="p", bins=np.linspace(0.0, 0.5, 21))

    print("Histograms of the probabilities of the detection algorithm by (label, true label)")
    pd.concat([p_df, p_df_n]).hist(by=["detected", "truth"],
                                   column="p",
                                   bins=np.linspace(0.0, 1.0, 21),
                                   sharex=True)


def get_correction_confidence_df(d, actual_errors):
    confidence_list = list(d.correction_confidences.items())
    is_correctly_detected = [cell_conf_tuple[0] in actual_errors for cell_conf_tuple in confidence_list]
    is_correctly_corrected = [cell_conf_tuple[0] in actual_errors and
                              d.corrected_cells[cell_conf_tuple[0]] == actual_errors[cell_conf_tuple[0]]
                              for cell_conf_tuple in confidence_list]
    return DataFrame({"cell": [item[0] for item in confidence_list],
                                          "confidence": [item[1] for item in confidence_list],
                                          "detection_correct": is_correctly_detected,
                                          "correct": is_correctly_corrected})


def correction_confidence_distributions(correction_confidence_df):
    print("Distribution of confidences for wrong (False) and correct (True) corrections:")

    correction_confidence_df.hist(by="correct", column="confidence", sharey=True, bins=np.linspace(0.5,1.0,21))


def correction_correctness_by_confidence(correction_confidence_df):
    print("Empirical probability of a correction being wrong given its confidence:")

    true_confidences = correction_confidence_df.loc[correction_confidence_df["correct"], "confidence"]
    false_confidences = correction_confidence_df.loc[~correction_confidence_df["correct"], "confidence"]
    evidence = correction_confidence_df.loc[:, "confidence"]

    evidence_hist, _ = np.histogram(evidence, bins=np.linspace(0.5, 1.0, 6))
    false_hist, _ = np.histogram(false_confidences, bins=np.linspace(0.5, 1.0, 6))
    true_hist, _ = np.histogram(true_confidences, bins=np.linspace(0.5, 1.0, 6))

    # print(evidence_hist)
    # print(false_hist)

    error_probability = np.divide(false_hist, np.where(evidence_hist == 0, 1, evidence_hist))
    correct_probability = np.divide(true_hist, np.where(evidence_hist == 0, 1, evidence_hist))
    # print(len(np.arange(0.5, 1.0, 0.05)))

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey="all", sharex="all")

    ax1.set_ylim([0.0, 1.0])

    ax1.bar(np.arange(0.5, 1.0, 0.1), error_probability, width=0.1, align="edge")
    ax2.bar(np.arange(0.5, 1.0, 0.1), correct_probability, width=0.1, align="edge")
    return fig


def build_change_dataframe(data: DataFrame, correction_dict: Dict[Tuple[int, int], any]) -> DataFrame:
    changes = []

    for cell, new_value in correction_dict.items():
        changes.append((cell[0], cell[1], data.iloc[cell], new_value, type(data.iloc[cell]), type(new_value)))

    return DataFrame(data=changes,
                     columns=["row", "column", "value_before", "value_after", "type_before", "type_after"])


def result_analysis(data: DataFrame, correction_dict: Dict[Tuple[int, int], any]):
    change_df: DataFrame = build_change_dataframe(data, correction_dict)
    number_of_rows = data.shape[0]

    if len(change_df.index) > 0:

        change_df["value_before_string"] = change_df["value_before"].astype(str)
        change_df["value_after_string"] = change_df["value_after"].astype(str)

        changed_attributes = [(i, data.columns[i]) for i in change_df["column"].unique().tolist()]
        number_of_changed_tuples = len(change_df["row"].unique())
        p_changed_tuples = number_of_changed_tuples / number_of_rows
        try:
            change_frequencies = change_df.value_counts(subset=["value_before", "value_after"])
        except Exception:
            change_frequencies = change_df.value_counts(subset=["value_before_string", "value_after_string"])
        relative_change_frequencies = change_frequencies / change_frequencies.sum()

        number_of_changes_per_column = change_df.value_counts(subset=["column"])
        p_changes_per_column = number_of_changes_per_column / number_of_changes_per_column.sum()

        types = change_df[["column", "type_before", "type_after"]].drop_duplicates()

        groups = change_df.groupby(["column", "value_after_string"])
        injectivity_test = groups["value_before_string"].agg("nunique") == 1

        string = ""
        string += f"Changed attributes: {changed_attributes}\n"
        string += f"Number of changed cells: {len(change_df.index)}\n"
        string += f"Number of changed tuples: {number_of_changed_tuples}\n"
        string += f"% of tuples changed: {p_changed_tuples*100:.2f}"
        string += "All changes:\n"
        string += change_df.__repr__() + "\n"
        string += "Type changes:\n"
        string += types.__repr__() + "\n"
        string += "Change frequency:\n"
        string += relative_change_frequencies.__repr__() + "\n"
        string += "Distribution of changes over columns:\n"
        string += p_changes_per_column.__repr__() + "\n"
        if injectivity_test.all():
            string += "All changes are injective (and therefore bijective)"
        else:
            string += "Not all changes are injective! look:\n"
            string += injectivity_test.__repr__()
        return string

    else:
        return "No Changes"


def print_feature_vector(v):
    print("[", end="")
    print(*[f"{feature:.2f}" for feature in v], sep=" ", end="")
    print("]")


def explain_detection(d, cell):
    print(f"Original Value: '{d.dataframe.iloc[cell]}'")
    print(f"Label: {int(cell in d.detected_cells)}")
    print(f"Confidence: {d.detected_cells[cell][1] if cell in d.detected_cells else d.undetected_cells[cell][0]}")
    k = max(d.cells_clusters_k_j_ce.keys())
    column = cell[1]
    cluster_id = d.cells_clusters_k_j_ce[k][column][cell]
    print(f"Cell belongs to cluster (Column {column}, Cluster {cluster_id})")
    print(f"Total number of cells in this cluster: {len(d.clusters_k_j_c_ce[k][column][cluster_id])}")

    labeled_cells = d.labels_per_cluster[(column, cluster_id)]

    if cell in d.extended_labeled_cells and not cell in d.labeled_cells:
        print(f"Cell was labeled by cluster label extension with label {d.extended_labeled_cells[cell]}")

        cluster_cells = list(d.clusters_k_j_c_ce[k][column][cluster_id].keys())
        cluster_cell_rows = [cluster_cell[0] for cluster_cell in cluster_cells]

        if len(cluster_cells) > 1:
            print("Other cells in the same cluster (sample of max 5 cells):")
            example_cells = sample(cluster_cells, min(len(cluster_cells), 5))
            for example_cell in example_cells:
                print(f"{example_cell}: '{d.dataframe.iloc[example_cell]}' Features: {d.column_features[column][example_cell[0], :]}")

            print("Mean of all feature vectors in the cluster and feature vector of the current cell for comparison:")
            print_feature_vector(d.column_features[column][cluster_cell_rows].mean(axis=0))
            print_feature_vector(d.column_features[column][cell[0]])
        else:
            print("This cell is the only cell in its cluster.")

        print("This cluster was labeled because of the user labels of the following cells in its cluster: ")
        for labeled_cell, label in labeled_cells.items():
                print(f"{labeled_cell}: '{d.dataframe.iloc[labeled_cell]}' Label: {label}")
    elif cell in d.labeled_cells:
        print(f"This cell was labeled by the user with label {d.labeled_cells[cell]}")
    else:
        print(f"This cell is not part of a labeled cluster. It was labeled by the classification.")

        if len(labeled_cells) > 0:
            print("However, there are the following labeled cells in its cluster:")
            for labeled_cell, label in labeled_cells.items():
                print(f"{labeled_cell}: '{d.dataframe.iloc[labeled_cell]}' Label: {label}")
        else:
            print("There are no labeled cells in its cluster.")


def alternative_corrections_overview(d, cell):
    print(f"Correction: {d.corrected_cells[cell]}")
    print(f"Correction confidence: {d.correction_confidences[cell]}")
    print(f"Alternative corrections and their confidence values: {d.correction_collection[cell]}")
    if d.correction_confidences[cell] < d.correction_collection[cell][d.corrected_cells[cell]]:
        print("Hint: The actual correction is included in the alternative corrections and the confidence value\n"
              "in the alternative corrections can be higher. This is because of baran prefering models trained with\n"
              "more examples. The alternative correction contains the maximum confidence for every repair value recorded\n"
              "over the cleaning process.")