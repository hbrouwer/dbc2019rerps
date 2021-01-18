#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Harm Brouwer <me@hbrouwer.eu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---- Last modified: January 2021, Harm Brouwer ----

import rerps.models as models

import matplotlib.pyplot as plt
import numpy as np

"""regression-based ERP estimation.
    
Minimal implementation of regression-based ERP (rERP) waveform estimation,
as proposed in:

Smith, N.J., Kutas, M., Regression-based estimation of ERP waveforms: I. The
    rERP framework, Psychophysiology, 2015, Vol. 52, pp. 157-168

Smith, N.J., Kutas, M., Regression-based estimation of ERP waveforms: II.
    Non-linear effects, overlap correction, and practical considerations,
    Psychophysiology, 2015, Vol. 52, pp. 169-181

This module implements plotting of (r)ERP waveforms and model coefficients.

"""

def plot_voltages(dsm, x, y, groupby, title=None, legend=True, ax=None, colors=None, ymin=None, ymax=None):
    """Plots voltages for a single electrode.

    Args:
        dsm (:obj:`DataSummary`):
            Summary of an Event-Related brain Potentials data set.
        x (:obj:`str`):
            name of the descriptor column that determines the x-axis
            (typically 'time').
        y (:obj:`str`):
            name of electrode to be plotted.
        groupby (:obj:`str`):
            name of the descriptor column that determines the grouping
            (typically 'condition').
        title (:obj:`str`):
            title of the graph
        legend (:obj:`bool`):
            flags whether a legend should be added.
        ax (:obj:`Axes`):
            axes.Axes object to plot to.
        colors (:obj:`list` of :obj:`str`):
            list of colors to use for plotting
        ymin (:obj:`float`):
            minimum of y axis
        ymax (:obj:`float`):
            maximum of y axis

    Returns:
        (:obj:`Figure`, optional): Figure
        (:obj:`Axes`): axes.Axes object.

    """
    newfig = False
    if (ax == None):
        newfig = True
        fig, ax = plt.subplots()
        ax.invert_yaxis()

    if (colors):
        ax.set_prop_cycle(color=colors)

    groups = np.unique(dsm.means[:,dsm.descriptors[groupby]])
    for g in groups:
        # means
        x_vals = dsm.means[dsm.means[:,
            dsm.descriptors[groupby]] == g,
            dsm.descriptors[x]]
        x_vals = x_vals.astype(float)
        y_vals = dsm.means[dsm.means[:,
            dsm.descriptors[groupby]] == g,
            dsm.electrodes[y]]
        y_vals = y_vals.astype(float)
        ax.plot(x_vals, y_vals, label=g)
        # CIs
        y_sems = dsm.sems[dsm.sems[:,
            dsm.descriptors[groupby]] == g,
            dsm.electrodes[y]]
        y_sems = y_sems.astype(float)
        y_lvals = y_vals - 2 * y_sems
        y_uvals = y_vals + 2 * y_sems
        ax.fill_between(x_vals, y_lvals, y_uvals, alpha=.2)

    ax.grid()
    ax.axvspan(300,500,  color="grey", alpha=0.2)
    ax.axvspan(600,1000, color="grey", alpha=0.2)
    ax.axhline(y=0, color="black")
    ax.axvline(x=0, color="black")
    
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    
    if (ymin and ymax):
        ax.set_ylim(ymin, ymax)
    
    if (legend):
        ax.legend(loc = "lower left", fontsize=14)
    if (title):
        ax.set_title(title, fontsize=16)

    if (newfig):
        return fig, ax
    else:
        return ax

def plot_voltages_grid(dsm, x, ys, groupby, title=None, colors=None, ymin=None, ymax=None):
    """Plots voltages for a grid of electrodes.

    Args:
        dsm (:obj:`DataSet`):
            Summary of an Event-Related brain Potentials data set.
        x (:obj:`str`):
            name of the descriptor column that determines the x-axis
            (typically 'time').
        ys (:obj:`list` of :obj:`str`):
            names of electrodes to be plotted.
        groupby (:obj:`str`):
            name of the descriptor column that determines the grouping
            (typically 'condition').
        title (:obj:`str`):
            global title of the graph
        colors (:obj:`list` of :obj:`str`):
            list of colors to use for plotting
        ymin (:obj:`float`):
            minimum of y axis
        ymax (:obj:`float`):
            maximum of y axis

    Returns:
        (:obj:`Figure`): Figure
        (:obj:`Axes`): axes.Axes object.

    """
    fig, axes = plt.subplots(len(ys), sharey=True)

    legend = False
    for i, y in enumerate(ys):
        if (i == len(ys) - 1):
            legend = True
            axes[i].invert_yaxis()
        plot_voltages(dsm, x, y, groupby, title=y, legend=legend, ax=axes[i], colors=colors, ymin=ymin, ymax=ymax)

    if (title):
        fig.suptitle(title, fontsize=18, x=.5, y=.95)
   
    return fig, axes

def plot_coefficients(msm, x, y, anchor=True, title=None, legend=True, ax=None, colors=None, ymin=None, ymax=None):
    """Plots coefficients for a single electrode.
    
    Args:
        msm (:obj:`DataSet`):
            Summary of a Linear regression coefficients set.
        x (:obj:`str`):
            name of the descriptor column that determines the x-axis
            (typically 'time').
        ys (:obj:`list` of :obj:`str`):
            names of electrodes to be plotted.
        anchor (:obj:`bool`):
            flags whether slopes should be anchored to the intercept.
        title (:obj:`str`):
            global title of the graph
        legend (:obj:`bool`):
            flags whether a legend should be added.
        ax (:obj:`Axes`):
            axes.Axes object to plot to.
        colors (:obj:`list` of :obj:`str`):
            list of colors to use for plotting
        ymin (:obj:`float`):
            minimum of y axis
        ymax (:obj:`float`):
            maximum of y axis

    Returns:
        (:obj:`Figure`, optional): Figure
        (:obj:`Axes`): axes.Axes object.
    
    """
    newfig = False
    if (ax == None):
        newfig = True
        fig, ax = plt.subplots()
        ax.invert_yaxis()

    if (colors):
        ax.set_prop_cycle(color=colors)
    
    for i, p in enumerate(msm.predictors):
        # means
        x_vals = msm.means[:,msm.descriptors[x]]
        x_vals = x_vals.astype(float)
        y_vals = msm.means[:,msm.coefficients[(y,p)]]
        y_vals = y_vals.astype(float)
        l = p
        if (anchor and i > 0):
            i_vals = msm.means[:,msm.coefficients[(y,msm.predictors[0])]]
            i_vals = i_vals.astype(float)
            y_vals = y_vals + i_vals
            l = msm.predictors[0] + " + " + p
        ax.plot(x_vals, y_vals, label=l)
        # CIs
        y_sems = msm.sems[:,msm.coefficients[(y,p)]]
        y_sems = y_sems.astype(float)
        y_lvals = y_vals - 2 * y_sems
        y_uvals = y_vals + 2 * y_sems
        ax.fill_between(x_vals, y_lvals, y_uvals, alpha=.2)

    ax.grid()
    ax.axvspan(300,500,  color="grey", alpha=0.2)
    ax.axvspan(600,1000, color="grey", alpha=0.2)
    ax.axhline(y=0, color="black")
    ax.axvline(x=0, color="black")
    
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    
    if (ymin and ymax):
        ax.set_ylim(ymin, ymax)
    
    if (legend):
        ax.legend(loc = "lower left", fontsize=14)
    if (title):
        ax.set_title(title, fontsize=16)

    if (newfig):
        return fig, ax
    else:
        return ax

def plot_coefficients_grid(msm, x, ys, anchor=True, title=None, colors=None, ymin=None, ymax=None):
    """Plots coefficients for a grid of electrodes.

    Args:
        msm (:obj:`DataSet`):
            Summary of a Linear regression coefficients set.
        x (:obj:`str`):
            name of the descriptor column that determines the x-axis
            (typically 'time').
        ys (:obj:`list` of :obj:`str`):
            names of electrodes to be plotted.
        anchor (:obj:`bool`):
            flags whether slopes should be anchored to the intercept.
        title (:obj:`str`):
            global title of the graph
        colors (:obj:`list` of :obj:`str`):
            list of colors to use for plotting
        ymin (:obj:`float`):
            minimum of y axis
        ymax (:obj:`float`):
            maximum of y axis

    Returns:
        (:obj:`Figure`): Figure
        (:obj:`Axes`): axes.Axes object.

    """
    fig, axes = plt.subplots(len(ys), sharey=True)

    legend = False
    for i, y in enumerate(ys):
        if (i == len(ys) - 1):
            legend = True
            axes[i].invert_yaxis()
        plot_coefficients(msm, x, y, anchor=anchor, title=y, legend=legend, ax=axes[i], colors=colors, ymin=ymin, ymax=ymax)

    if (title):
        fig.suptitle(title, fontsize=18, x=.5, y=.95)
   
    return fig, axes
