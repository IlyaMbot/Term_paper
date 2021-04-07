#! /usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def file_read_column(path, number):

    df = pd.read_table(path, header = None, sep = r"\s+")
    return(df.iloc[:,number])


def general_plot(time1, time2, data1, data2):
    lsize = 30
    
    fig, ax = plt.subplots(1, 1)
    
    plt.tight_layout()

    ax.plot(time1, data1, "bo", label = "Model")
    ax.plot(time2, data2, label = "Solar flare 304A SDO/AIA", linewidth = 3.0)
    
    y = np.arange(0, 1.1, 0.1)
    x = np.arange(-5, 15, 1)

    
    plt.yticks(y)
    plt.xticks(x)

    ax.set_title("Temperature profile", size = lsize)
    ax.set_xlabel('Time', size = lsize)
    

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize - 5)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize - 5)

    ax.axis([-5, 14.5, -0.1, 1.1])

    ax.tick_params(axis = 'x', rotation = 45)
    ax.grid()
    ax.legend(fontsize = lsize - 2)
    
    plt.show()



alpha = np.float(-0.9)

data = file_read_column("prof_304A.dat", 1)
time2 = file_read_column("prof_304A.dat", 0)

def model_rand(t):
    
    delta_rand = np.random.default_rng().normal(loc = 0.0, scale = 0.01)
    delta_rand2 = np.random.default_rng().normal(loc = 0.0, scale = 0.1)

    alpha = np.float(-0.9)
    sigma = 2
    T = np.exp( (t - delta_rand2) * sigma) + delta_rand
    size_const_rad = np.float( (1 - (0.5)**(alpha))/alpha )
    
    if(t >= 0):
        return( (1 - alpha * (t - delta_rand2) * size_const_rad )**(1/alpha) + delta_rand)
    return(T)

model = []

time1 = np.arange(-6, 15, 0.1)

for i in time1:
    model.append( model_rand(i))

#print(model)
general_plot(time1, time2, model, data)