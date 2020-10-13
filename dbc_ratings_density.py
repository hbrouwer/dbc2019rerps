#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Density plots for association and plausibility ratings, as reported in:
#
# Brouwer, H., Delogu, F., and Crocker, M. W. (2020). Splitting
#   Event‐Related Potentials: Modeling Latent Components using
#   Regression‐based Waveform Estimation. European Journal of Neuroscience.
#   doi: 10.1111/ejn.14961
#
# 10/09/20: Harm Brouwer <me@hbrouwer.eu>

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/dbc_ratings.csv", sep=",")

df["Condition"] = df["Condition"].replace("control", "baseline")
df["Condition"] = df["Condition"].replace("script-related", "event-related")
df["Condition"] = df["Condition"].replace("script-unrelated", "event-unrelated")

df = df.set_index("Condition")

        #####################
        #### association ####
        #####################

fig, ax = plt.subplots()
ax.set_prop_cycle(color = ["#1f77b4", "#2ca02c"])
for c in ["baseline", "event-unrelated"]:
    ax.hist(df.loc[c]["Assoc"], alpha=.75, bins=14)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="both", which="minor", labelsize=11)
    ax.set_title("Association", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_xlabel("Rating", fontsize=14)
    ax.set_ylim(0,30)
    ax.set_xlim(1,7)
ax.legend(["baseline / event-related", "event-unrelated"])

print("\n[ figures/dbc_assoc_ratings.pdf ]\n")
fig.set_size_inches(5, 5)
fig.set_tight_layout(True)
fig.savefig("figures/dbc_assoc_ratings.pdf")

        ######################
        #### plausibility ####
        ######################

fig, ax = plt.subplots()
ax.set_prop_cycle(color=["#1f77b4", "#ff7f0e", "#2ca02c"])
for c in ["baseline", "event-related", "event-unrelated"]:
    ax.hist(df.loc[c]["Plaus"], alpha=.75, bins=14)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="both", which="minor", labelsize=11)
    ax.set_title("Plausibility", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_xlabel("Rating", fontsize=14)
    ax.set_ylim(0,30)
    ax.set_xlim(1,7)
ax.legend(["baseline", "event-related", "event-unrelated"])

print("\n[ figures/dbc_plaus_ratings.pdf ]\n")
fig.set_size_inches(5, 5)
fig.set_tight_layout(True)
fig.savefig("figures/dbc_plaus_ratings.pdf")
