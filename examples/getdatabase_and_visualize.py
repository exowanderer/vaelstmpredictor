import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import statsmodels.robust.scale as scale


parser = argparse.ArgumentParser()
parser.add_argument('--getdatabase', action='store_true', help='force new download of database')
parser.add_argument('--filename', type=str, help='filename to load/save the dataframe')
parser.add_argument('--robust', action='store_true', help='Display robust percentiles')
clargs = parser.parse_args()

def plot_mpl_colors():
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k, color in enumerate(color_cycle):
        plt.plot([k,k+1], [k,k+1], color=color, lw=5)
        plt.annotate(str(k), [k+0.25, k+0.75], color=color, fontsize=20)
    plt.axis('off')
    plt.tight_layout()

def plot_bokeh_candelstick(df):
    from math import pi
    from bokeh.plotting import figure, show, output_file

    inc = df.close > df.open
    dec = df.open > df.close
    w = 12*60*60*1000 # half day in ms

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, title = "MSFT Candlestick")
    p.xaxis.major_label_orientation = pi/4
    p.grid.grid_line_alpha=0.3

    p.segment(df.date, df.high, df.date, df.low, color="black")
    p.vbar(df.date[inc], w, df.open[inc], df.close[inc], fill_color="#D5E1DD", line_color="black")
    p.vbar(df.date[dec], w, df.open[dec], df.close[dec], fill_color="#F2583E", line_color="black")

    output_file("candlestick.html", title="candlestick.py example")

    show(p)  # open a browser

def plot_fitness(df, xlabel='generationID', ylabel='fitness', nSig=5, percentiles=[32., 68], robust=False):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fitnesses = df[ylabel][df.isTrained == 2]
    genIDs = df[xlabel].unique()
    
    med_fitness = np.median(fitnesses)
    mad_fitness = scale.mad(fitnesses)
    
    best_per_gen = np.zeros(len(genIDs))
    best_running = np.zeros(len(genIDs))

    worst_per_gen = np.zeros(len(genIDs))
    worst_running = np.zeros(len(genIDs))

    med_per_gen = np.zeros(len(genIDs))
    percentiles_per_gen = np.zeros((len(genIDs), len(percentiles)))

    for gen in genIDs:
        fitness_gen = fitnesses[df[xlabel] == gen].values
        plt.plot([gen]*len(fitness_gen), fitness_gen, 'o', color='grey', alpha=0.5, mew=0)

        best_per_gen[gen] = np.max(fitness_gen)
        best_running[gen] = np.max([best_per_gen[gen], np.max(best_running)])
        
        worst_per_gen[gen] = np.min(fitness_gen)
        worst_running[gen] = np.max([worst_per_gen[gen], np.max(worst_running)])

        med_per_gen[gen] = np.median(fitness_gen)

        if robust:
            std_ = np.std(fitness_gen)
            fitness_gen = fitness_gen[abs(fitness_gen - med_per_gen[gen]) < std_]
        
        percentiles_per_gen[gen] = np.percentile(fitness_gen, percentiles)
    
    plt.plot([],[], 'o', color='grey', alpha=0.5, mew=0, label='All Fitness Values')
    plt.plot(genIDs, best_per_gen, 'o-', color=color_cycle[3], alpha=0.8, label='Best per Gen')
    plt.plot(genIDs, best_running, 'o-', color=color_cycle[1], alpha=0.8, label='Best Running')

    plt.plot(genIDs, worst_per_gen, 'o-', color=color_cycle[0], alpha=0.8, label='Worst per Gen')
    plt.plot(genIDs, worst_running, 'o-', color=color_cycle[2], alpha=0.8, label='Worst Running')
    
    plt.plot(genIDs, med_per_gen, '--', color='black', lw=3, alpha=0.75, label='Median per Gen')

    for k in range(len(percentiles)//2):
        plt.fill_between(genIDs, percentiles_per_gen.T[k], percentiles_per_gen.T[-(k+1)],
                         color='grey', alpha=0.25)

    extra = {True:'Robust ', False:''}[robust]
    label = '{}Width of Distro per Gen'.format(extra)
    plt.fill_between([],[],color='grey', alpha=0.25, label=label)

    # plt.fill_between(genIDs, percentiles_per_gen.T[0]-med_per_gen+best_running, 
    #                      percentiles_per_gen.T[-1]-med_per_gen+best_running,
    #                      color='grey', alpha=0.25)

    std_fitness = np.std(fitnesses)
    label_botb = 'Best +/- 1-sigma: {} +/- {}'.format(np.max(best_running), np.round(std_fitness,1))
    plt.axhline(np.max(best_running), ls='-', alpha=0.25, lw=std_fitness*5, label=label_botb)
    # plt.annotate(str(np.max(best_running)), (max(genIDs), np.max(best_running)+std_fitness*0.05))
    
    plt.ylim(np.min(worst_per_gen) - mad_fitness, np.max(best_running) + mad_fitness)
    plt.title('Genetic Algorithm Fitness over Generations', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.xticks(genIDs)
    plt.legend(loc=0, fontsize=15)

    ax = plt.gcf().get_axes()[0]
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.subplots_adjust(top=0.953, bottom=0.082, left=0.046, right=0.990)
    plt.show()

if clargs.filename is None:
    csv_filename = 'philippesaade11_pythonanywhere_GetDatabase.csv'
else:
    csv_filename = clargs.filename

if os.path.exists(csv_filename) and not clargs.getdatabase:
    df = pd.read_csv(csv_filename)
else:
    url = r'http://philippesaade11.pythonanywhere.com/GetDatabase'
    r = requests.get(url)
    df = pd.DataFrame(json.loads(r.text))
    df.to_csv(csv_filename, index=False)

percentiles=[0.3, 5., 32., 68, 95, 99.7]
# percentiles=[32., 38, 44, 56, 62, 68]
plot_fitness(df, percentiles=percentiles, robust=clargs.robust)