import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt


metrics_dict = {'Delay': 4, 'Rush Hour Delay': 6, 'Low Hour Delay': 7, 'Max Delay': 8, 'Max Queue': 9}


def plot(df, label, metric, fill=False):
    x, y, yerr = [], [], []

    if label is not 'CMAES':
        for ep_num in range(40):
            if label == 'Actuated':
                eps = 0
            else:
                eps = ep_num
            episode = df.loc[df[1] == eps].to_numpy()[:, metrics_dict[metric]]
            mean = np.mean(episode)
            interval = st.t.interval(0.95, len(episode), loc=mean, scale=st.sem(episode))[1]
            x.append(ep_num+1)
            y.append(mean)
            yerr.append(interval - mean)
    else:
        episode = df.loc[df[1] == 0].to_numpy()[:, metrics_dict[metric]]
        mean = np.mean(episode)
        interval = st.t.interval(0.95, len(episode), loc=mean, scale=st.sem(episode))[1]
        x.append(45)
        y.append(mean)
        yerr.append(interval - mean)
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    if fill:
        plt.plot(x, y, label=label)
        plt.fill_between(x, y - yerr, y + yerr)
    else:
        plt.errorbar(x, y, label=label, yerr=yerr, capsize=3)
        points = np.asarray([1,5,10,15,20,25,30,35,40,45])
        labels = ('1','5','10','15','20','25','30','35','40','..3240')  # 3240, 3600, 3984
        plt.xticks(points, labels)
    return y, yerr


demand = ['Med']    # 'Med', 'High']
methods = ['Actuated', 'DQN', 'CMAES', 'RPPO', 'DRSQ', 'DRHQ']    # 'DRQ'
metrics = ['Delay']     # 'Rush Hour Delay', 'Low Hour Delay', 'Max Delay', 'Max Queue']
plt.rc('axes', labelsize=22, titlesize=22)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
for met in metrics:
    print(met)
    for dem in demand:
        min, max = np.inf, -np.inf
        for meth in methods:
            try :
                data = pd.read_csv(dem+meth+'.csv', header=None)
                print(dem+meth)
            except FileNotFoundError:
                print('Skipping', dem+meth)
                continue
            y, yerr = plot(data, meth, met, fill=False)
            ymin = np.min(y[:] - yerr[:])
            if ymin < min:
                min = ymin
            if meth is not 'CMAES':
                ymax = np.max(y[9:]+yerr[9:])
                if ymax > max:
                    max = ymax
        plt.title(dem+' Demand')
        plt.xlabel('Episode')
        #plt.ylabel(met)
        plt.rc('legend', fontsize=25)
        plt.legend()
        plt.ylim(min * 0.98, max * 1.1)
        plt.savefig(dem+met+'.png')
        plt.show()
        print()
