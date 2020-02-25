import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt


metrics_dict = {'Delay': 2}


def plot(df, label, metric, fill=False):
    x, y, yerr = [], [], []
    for ep_num in range(200):
        if meth == 'Actuated' or meth == 'DQN':
            eps = 0
        else:
            eps = ep_num
        episode = df.loc[df[1] == eps].to_numpy()[:, metrics_dict[metric]]
        mean = np.mean(episode)
        #interval = st.t.interval(0.95, len(episode), loc=mean, scale=st.sem(episode))[1]
        x.append(ep_num+1)
        y.append(mean)
        yerr.append(0)
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    plt.plot(x, y, label=label)
    return y, yerr


demand = ['CMAES']
methods = ['Actuated', 'Regulatable', 'DQN']
metrics = ['Delay']
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
            ymin = np.min(y[25:]-yerr[25:])
            ymax = np.max(y[25:]+yerr[25:])
            if ymin < min:
                min = ymin
            if ymax > max:
                max = ymax
        plt.xlabel('Epoch (24 episodes)')
        plt.ylabel(met)
        plt.legend(fontsize=16)
        plt.ylim(min * 0.9, max * 1.1)
        plt.savefig(dem+met+'.png')
        plt.show()
        print()
