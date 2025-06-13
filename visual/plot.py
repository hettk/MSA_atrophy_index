import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels import graphics
from scipy import stats
from matplotlib import pyplot as plt, markers, colors
from seaborn import violinplot as sns_violin


Colors = {"Blue":"#002349",
          "Green":"#52A400",
          "Orange":"#492500",
          "Cyan":"#00A4A4",
          "Red":"#490000",
          "IndianRed":"#C0392B",
          "DirtyRed":"#922B21",
          "Salmon":"#FA8072",
          "Dark":"#333333"}

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')
    

def partial_regression_plot(mdl, exog, grid, ax, color, ci_display=True, label=""):
    try:
        prediction = mdl.get_prediction(grid)
    except:
        prediction = mdl.predict(grid)
    endog_pred = prediction.summary_frame()["mean"].to_numpy()

    if ci_display:
        ci_low = prediction.summary_frame()["mean_ci_lower"].to_numpy()
        ci_up = prediction.summary_frame()["mean_ci_upper"].to_numpy()
        ci = np.concatenate([ci_low, np.flip(ci_up)])
        avg_tr = np.concatenate([exog, np.flip(exog)])
        ax.fill(avg_tr,ci, alpha=0.4, color=color)

    ax.plot(exog, endog_pred, color=color, linewidth=2, label=label)


# def plot_significance(ax, bbox, height, stats):
#     ax.hline(height, bbox[0], bbox[1], color='black', linestyle='-', linewidths=2.5)
#     ax.vline(bbox[0], height, height-)

def violin_plot(data, ax, location, color, width=0.15, scatter=True, violin=True, scat_offset=0):
    if violin:
        vp_ = ax.violinplot(data, [location], widths=width, showmeans=False, showmedians=False,
            showextrema=False)
        for pc in vp_['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.4)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
    whiskers_min, whiskers_max = np.min(data), np.max(data)

    # Scatter plot
    if scatter:
        idx = np.argsort(data, axis=None)
        data = data[idx]
        rseed = np.random.normal(0, width/10, data.shape)
        ax.scatter(location+rseed+scat_offset, data, marker=markers.MarkerStyle('o', fillstyle='full'),
                   color=color, s=30, zorder=-303, alpha=0.7)
        ax.scatter(location+rseed+scat_offset, data, marker=markers.MarkerStyle('o', fillstyle='full'),
                   color='w', s=30, zorder=-303, alpha=0.2)
    else:
        idx = np.arange(0,len(data))
        rseed = np.zeros(data.shape)
    radius = width/10

    # Box plot
    ax.vlines(location, whiskers_min, whiskers_max, color=color, linestyle='-', linewidths=1)

    ax.fill([location-radius, location+radius, location+radius, location-radius],
            [quartile1, quartile1, quartile3, quartile3], color=color, edgecolor=color, alpha=1)
    ax.hlines(medians,location-radius, location+radius, color='w', linestyle='-', linewidths=2.5)
    return location+rseed[np.argsort(idx)]+scat_offset


def box_right(data, ax, location, color, width=0.15, alpha=1):
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
    whiskers_min, whiskers_max = np.min(data), np.max(data)
    radius = width/10
    ax.vlines(location, whiskers_min, whiskers_max, color=color, linestyle='-', linewidths=1)
    ax.fill([location-radius, location, location, location-radius],
            [quartile1, quartile1, quartile3, quartile3], color=color, edgecolor=color, alpha=alpha)
    ax.hlines(medians,location-radius, location, color='w', linestyle='-', linewidths=2.5)
    return whiskers_max

def box_left(data, ax, location, color, width=0.15, alpha=1):
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
    whiskers_min, whiskers_max = np.min(data), np.max(data)
    radius = width/10
    ax.vlines(location, whiskers_min, whiskers_max, color=color, linestyle='-', linewidths=1)
    ax.fill([location, location+radius, location+radius, location],
            [quartile1, quartile1, quartile3, quartile3], color=color, edgecolor=color, alpha=alpha)
    ax.hlines(medians,location, location+radius, color='w', linestyle='-', linewidths=2.5)
    return whiskers_max

def scatter_right(data,ax,location,color,width=0.15):
    rseed = np.abs(np.random.normal(0, width, data.shape))
    ax.scatter(location - rseed, data, marker=markers.MarkerStyle('o', fillstyle='full'),
               color=color, s=30, zorder=3, alpha=0.7)
    ax.scatter(location - rseed, data, marker=markers.MarkerStyle('o', fillstyle='full'),
               color='w', s=30, zorder=3, alpha=0.2)

def scatter_left(data,ax,location,color,width=0.15):
    rseed = np.abs(np.random.normal(0, width, data.shape))
    ax.scatter(location + rseed, data, marker=markers.MarkerStyle('o', fillstyle='full'),
               color=color, s=30, zorder=3, alpha=0.7)
    ax.scatter(location + rseed, data, marker=markers.MarkerStyle('o', fillstyle='full'),
               color='w', s=30, zorder=3, alpha=0.2)


def violin_plots(data, ax, locations, colors, width=0.15, scatter=True, pvalready=False):
    vmax = np.zeros(len(locations))
    for i in range(len(locations)):
        vmax[i] = violin_plot(data[i], ax, locations[i], colors[i], width=width, scatter=scatter)



def bland_altman_plot(data1, data2, ax, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    ax.scatter(mean, diff, *args, **kwargs)
    ax.axhline(md,           color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='red', linestyle='--')
    ax.axhline(md - 1.96*sd, color='red', linestyle='--')