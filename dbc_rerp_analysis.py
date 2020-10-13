#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Full regression-based Waveform Estimation (rERP) analysis reported in:
#
# Brouwer, H., Delogu, F., and Crocker, M. W. (2020). Splitting
#   Event‐Related Potentials: Modeling Latent Components using
#   Regression‐based Waveform Estimation. European Journal of Neuroscience.
#   doi: 10.1111/ejn.14961
#
# 10/09/20: Harm Brouwer <me@hbrouwer.eu>

import rerps.models
import rerps.plots

import numpy as np
import pandas as pd

def generate():
    obs_data = rerps.models.DataSet(
        filename    = "data/dbc_data.csv",
        descriptors = ["Subject", "Timestamp", "Condition", "ItemNum"],
        electrodes  = ["Fz", "Cz", "Pz", "F3", "FC1", "FC5", "F4", "FC2", "FC6",
                       "P3", "CP1", "CP5", "P4", "CP2", "CP6", "O1", "Oz", "O2"],
        predictors  = ["Plaus","Assoc"])

    obs_data.rename_descriptor_level("Condition", "control",          "baseline")
    obs_data.rename_descriptor_level("Condition", "script-related",   "event-related")
    obs_data.rename_descriptor_level("Condition", "script-unrelated", "event-unrelated")

    obs_data.rename_predictor("Plaus", "plausibility")
    obs_data.rename_predictor("Assoc", "association")

    obs_data.invert_predictor("plausibility", 7.0)
    obs_data.invert_predictor("association", 7.0)

    obs_data.zscore_predictor("plausibility")
    obs_data.zscore_predictor("association")

        ####################
        #### potentials ####
        ####################

    print("\n[ figures/dbc_potentials.pdf ]\n")
    obs_data_summary = rerps.models.DataSummary(obs_data, ["Condition", "Subject", "Timestamp"])
    obs_data_summary = rerps.models.DataSummary(obs_data_summary, ["Condition", "Timestamp"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = rerps.plots.plot_voltages_grid(obs_data_summary, "Timestamp", ["Fz", "Cz", "Pz"],
            "Condition", title="Event-Related Potentials", colors=colors)
    fig.set_size_inches(15, 10)
    fig.savefig("figures/dbc_potentials.pdf", bbox_inches='tight')

    print("\n[ stats/dbc_potentials_300-500.csv ]\n")
    time_window_averages(obs_data, 300, 500 ).to_csv("stats/dbc_potentials_300-500.csv",  index=False)
    print("\n[ stats/dbc_potentials_600-1000.csv ]\n")
    time_window_averages(obs_data, 600, 1000).to_csv("stats/dbc_potentials_600-1000.csv", index=False)
    print("\n[ stats/dbc_potentials_700-1000.csv ]\n")
    time_window_averages(obs_data, 700, 1000).to_csv("stats/dbc_potentials_700-1000.csv", index=False)
    print("\n[ stats/dbc_potentials_800-1000.csv ]\n")
    time_window_averages(obs_data, 800, 1000).to_csv("stats/dbc_potentials_800-1000.csv", index=False)

        ########################
        #### intercept-only ####
        ########################

    print("\n[ figures/dbc_intercept_est.pdf ]\n")
    models = rerps.models.regress(obs_data, ["Subject", "Timestamp"], [])
    est_data = rerps.models.estimate(obs_data, models)
    # isolate baseline, and rename
    est_data0 = est_data.copy()
    est_data0.array = est_data0.array[est_data0.array[:,est_data0.descriptors["Condition"]] == "baseline",:]
    est_data0.rename_descriptor_level("Condition", "baseline", "baseline / event-related / event-unrelated")
    est_data0_summary = rerps.models.DataSummary(est_data0, ["Condition", "Subject", "Timestamp"])
    est_data0_summary = rerps.models.DataSummary(est_data0_summary, ["Condition", "Timestamp"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = rerps.plots.plot_voltages_grid(est_data0_summary, "Timestamp", ["Fz", "Cz", "Pz"],
            "Condition", title="regression-based Event-Related Potentials", colors=colors)
    fig.set_size_inches(15, 10)
    fig.savefig("figures/dbc_intercept_est.pdf", bbox_inches='tight')

    print("\n[ figures/dbc_intercept_res.pdf ]\n")
    res_data = rerps.models.residuals(obs_data, est_data)
    res_data_summary = rerps.models.DataSummary(res_data, ["Condition", "Subject", "Timestamp"])
    res_data_summary = rerps.models.DataSummary(res_data_summary, ["Condition", "Timestamp"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = rerps.plots.plot_voltages_grid(res_data_summary, "Timestamp", ["Fz", "Cz", "Pz"],
            "Condition", title="Residuals", colors=colors)
    fig.set_size_inches(15, 10)
    fig.savefig("figures/dbc_intercept_res.pdf", bbox_inches='tight')

        ######################
        #### plausibility ####
        ######################

    print("\n[ figures/dbc_plaus_est.pdf ]\n")
    models = rerps.models.regress(obs_data, ["Subject", "Timestamp"], ["plausibility"])
    est_data = rerps.models.estimate(obs_data, models)
    est_data_summary = rerps.models.DataSummary(est_data, ["Condition", "Subject", "Timestamp"])
    est_data_summary = rerps.models.DataSummary(est_data_summary, ["Condition", "Timestamp"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = rerps.plots.plot_voltages_grid(est_data_summary, "Timestamp", ["Fz", "Cz", "Pz"],
            "Condition", title="regression-based Event-Related Potentials", colors=colors)
    fig.set_size_inches(15, 10)
    fig.savefig("figures/dbc_plaus_est.pdf", bbox_inches='tight')

    print("\n[ figures/dbc_plaus_res.pdf ]\n")
    res_data = rerps.models.residuals(obs_data, est_data)
    res_data_summary = rerps.models.DataSummary(res_data, ["Condition", "Subject", "Timestamp"])
    res_data_summary = rerps.models.DataSummary(res_data_summary, ["Condition", "Timestamp"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = rerps.plots.plot_voltages_grid(res_data_summary, "Timestamp", ["Fz", "Cz", "Pz"],
            "Condition", title="Residuals", colors=colors)
    fig.set_size_inches(15, 10)
    fig.savefig("figures/dbc_plaus_res.pdf", bbox_inches='tight')

        #####################
        #### association ####
        #####################

    print("\n[ figures/dbc_assoc_est.pdf ]\n")
    models = rerps.models.regress(obs_data, ["Subject", "Timestamp"], ["association"])
    est_data = rerps.models.estimate(obs_data, models)
    est_data_summary = rerps.models.DataSummary(est_data, ["Condition", "Subject", "Timestamp"])
    est_data_summary = rerps.models.DataSummary(est_data_summary, ["Condition", "Timestamp"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = rerps.plots.plot_voltages_grid(est_data_summary, "Timestamp", ["Fz", "Cz", "Pz"],
            "Condition", title="regression-based Event-Related Potentials", colors=colors)
    fig.set_size_inches(15, 10)
    fig.savefig("figures/dbc_assoc_est.pdf", bbox_inches='tight')

    print("\n[ figures/dbc_assoc_res.pdf ]\n")
    res_data = rerps.models.residuals(obs_data, est_data)
    res_data_summary = rerps.models.DataSummary(res_data, ["Condition", "Subject", "Timestamp"])
    res_data_summary = rerps.models.DataSummary(res_data_summary, ["Condition", "Timestamp"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = rerps.plots.plot_voltages_grid(res_data_summary, "Timestamp", ["Fz", "Cz", "Pz"],
            "Condition", title="Residuals", colors=colors)
    fig.set_size_inches(15, 10)
    fig.savefig("figures/dbc_assoc_res.pdf", bbox_inches='tight')

        ####################################
        #### plausibility + association ####
        ####################################

    print("\n[ figures/dbc_plaus+assoc_est.pdf ]\n")
    models = rerps.models.regress(obs_data, ["Subject", "Timestamp"], ["plausibility", "association"])
    est_data = rerps.models.estimate(obs_data, models)
    est_data_summary = rerps.models.DataSummary(est_data, ["Condition", "Subject", "Timestamp"])
    est_data_summary = rerps.models.DataSummary(est_data_summary, ["Condition", "Timestamp"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = rerps.plots.plot_voltages_grid(est_data_summary, "Timestamp", ["Fz", "Cz", "Pz"],
            "Condition", title="regression-based Event-Related Potentials", colors=colors)
    fig.set_size_inches(15, 10)
    fig.savefig("figures/dbc_plaus+assoc_est.pdf", bbox_inches='tight')

    print("\n[ stats/dbc_plaus+assoc_300-500.csv ]\n")
    time_window_averages(est_data, 300, 500 ).to_csv("stats/dbc_plaus+assoc_300-500.csv",  index=False)
    print("\n[ stats/dbc_plaus+assoc_600-1000.csv ]\n")
    time_window_averages(est_data, 600, 1000).to_csv("stats/dbc_plaus+assoc_600-1000.csv", index=False)
    print("\n[ stats/dbc_plaus+assoc_700-1000.csv ]\n")
    time_window_averages(est_data, 700, 1000).to_csv("stats/dbc_plaus+assoc_700-1000.csv", index=False)
    print("\n[ stats/dbc_plaus+assoc_800-1000.csv ]\n")
    time_window_averages(est_data, 800, 1000).to_csv("stats/dbc_plaus+assoc_800-1000.csv", index=False)

    print("\n[ figures/dbc_plaus+assoc_res.pdf ]\n")
    res_data = rerps.models.residuals(obs_data, est_data)
    res_data_summary = rerps.models.DataSummary(res_data, ["Condition", "Subject", "Timestamp"])
    res_data_summary = rerps.models.DataSummary(res_data_summary, ["Condition", "Timestamp"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = rerps.plots.plot_voltages_grid(res_data_summary, "Timestamp", ["Fz", "Cz", "Pz"],
            "Condition", title="Residuals", colors=colors, ymin=-2, ymax=2)
    fig.set_size_inches(15, 10)
    fig.savefig("figures/dbc_plaus+assoc_res.pdf", bbox_inches='tight')

    print("\n[ figures/dbc_plaus+assoc_coef.pdf ]\n")
    models_summary = rerps.models.ModelSummary(models, ["Timestamp"])
    colors = ["#d62728", "#9467bd", "#8c564b"]
    fig, ax = rerps.plots.plot_coefficients_grid(models_summary, "Timestamp", ["Fz", "Cz", "Pz"],
            anchor=True, title="Coefficients", colors=colors)
    fig.set_size_inches(15, 10)
    fig.savefig("figures/dbc_plaus+assoc_coef.pdf", bbox_inches='tight')

    print("\n[ figures/dbc_plaus0+assoc_est.pdf ]\n")
    obs_data0 = obs_data.copy()
    obs_data0.array[:,obs_data0.predictors["plausibility"]] = 0
    est_data = rerps.models.estimate(obs_data0, models)
    est_data_summary = rerps.models.DataSummary(est_data, ["Condition", "Subject", "Timestamp"])
    est_data_summary = rerps.models.DataSummary(est_data_summary, ["Condition", "Timestamp"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = rerps.plots.plot_voltages_grid(est_data_summary, "Timestamp", ["Fz", "Cz", "Pz"],
            "Condition", title="regression-based Event-Related Potentials", colors=colors)
    fig.set_size_inches(15, 10)
    fig.savefig("figures/dbc_plaus0+assoc_est.pdf", bbox_inches='tight')

    print("\n[ stats/dbc_plaus0+assoc_300-500.csv ]\n")
    time_window_averages(est_data, 300, 500 ).to_csv("stats/dbc_plaus0+assoc_300-500.csv",  index=False)
    print("\n[ stats/dbc_plaus0+assoc_600-1000.csv ]\n")
    time_window_averages(est_data, 600, 1000).to_csv("stats/dbc_plaus0+assoc_600-1000.csv", index=False)
    print("\n[ stats/dbc_plaus0+assoc_700-1000.csv ]\n")
    time_window_averages(est_data, 700, 1000).to_csv("stats/dbc_plaus0+assoc_700-1000.csv", index=False)
    print("\n[ stats/dbc_plaus0+assoc_800-1000.csv ]\n")
    time_window_averages(est_data, 800, 1000).to_csv("stats/dbc_plaus0+assoc_800-1000.csv", index=False)

    print("\n[ figures/dbc_plaus+assoc0_est.pdf ]\n")
    obs_data0 = obs_data.copy()
    obs_data0.array[:,obs_data0.predictors["association"]] = 0
    est_data = rerps.models.estimate(obs_data0, models)
    est_data_summary = rerps.models.DataSummary(est_data, ["Condition", "Subject", "Timestamp"])
    est_data_summary = rerps.models.DataSummary(est_data_summary, ["Condition", "Timestamp"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = rerps.plots.plot_voltages_grid(est_data_summary, "Timestamp", ["Fz", "Cz", "Pz"],
            "Condition", title="regression-based Event-Related Potentials", colors=colors)
    fig.set_size_inches(15, 10)
    fig.savefig("figures/dbc_plaus+assoc0_est.pdf", bbox_inches='tight')

    print("\n[ stats/dbc_plaus+assoc0_300-500.csv ]\n")
    time_window_averages(est_data, 300, 500 ).to_csv("stats/dbc_plaus+assoc0_300-500.csv",  index=False)
    print("\n[ stats/dbc_plaus+assoc0_600-1000.csv ]\n")
    time_window_averages(est_data, 600, 1000).to_csv("stats/dbc_plaus+assoc0_600-1000.csv", index=False)
    print("\n[ stats/dbc_plaus+assoc0_700-1000.csv ]\n")
    time_window_averages(est_data, 700, 1000).to_csv("stats/dbc_plaus+assoc0_700-1000.csv", index=False)
    print("\n[ stats/dbc_plaus+assoc0_800-1000.csv ]\n")
    time_window_averages(est_data, 800, 1000).to_csv("stats/dbc_plaus+assoc0_800-1000.csv", index=False)

###########################################################################
###########################################################################

def time_window_averages(ds, start, end):
    ts_idx = ds.descriptors["Timestamp"]
    sds = ds.copy()
    sds.array = sds.array[(sds.array[:,ts_idx] >= start) & (sds.array[:,ts_idx] < end),:]
    sds_summary = rerps.models.DataSummary(sds, ["Condition", "Subject"])

    nrows = sds_summary.means.shape[0] * len(sds_summary.electrodes)
    sds_lf = np.empty((nrows, 4), dtype=object)

    sds_idx = 0
    for idx in range(0, sds_summary.means.shape[0]):
        c = sds_summary.means[idx, sds_summary.descriptors["Condition"]]
        s = sds_summary.means[idx, sds_summary.descriptors["Subject"]]
        for e, i in sds_summary.electrodes.items():
            sds_lf[sds_idx,:] = [c, s, e, sds_summary.means[idx,i]]
            sds_idx = sds_idx + 1

    return pd.DataFrame(sds_lf, columns=["cond", "subject", "ch", "eeg"])

###########################################################################
###########################################################################

if __name__ == "__main__":
    generate()
