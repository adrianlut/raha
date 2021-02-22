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

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.bar(np.arange(0.5, 1.0, 0.1), error_probability, width=0.1, align="edge")
    ax2.bar(np.arange(0.5, 1.0, 0.1), correct_probability, width=0.1, align="edge")
    return fig
