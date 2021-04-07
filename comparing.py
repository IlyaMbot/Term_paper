#! /usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def file_read_column(path, number):

    df = pd.read_table(path, header = None, sep = r"\s+")
    return(df.iloc[:,number])


def general_plot(time, data1, data2, data3):
    lsize = 30
    
    fig, ax = plt.subplots(1, 1)
    
    plt.tight_layout()

    ax.plot(time, data1, label = "Conductive cooling", linewidth = 3.0)
    ax.plot(time, data2, label = "Radiative cooling", linewidth = 3.0)
    ax.plot(time, data3, label = "Solar flare 304A SDO/AIA", linewidth = 3.0)
    
    y = np.arange(0, 1.1, 0.1)
    x = np.arange(0, 5.5, 0.5)

    
    plt.yticks(y)
    plt.xticks(x)

    ax.set_title("Temperature profile", size = lsize)
    ax.set_xlabel('Time', size = lsize)
    

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize - 5)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize - 5)

    ax.axis([-0.1, 5, -0.1, 1.1])

    ax.tick_params(axis = 'x', rotation = 45)
    ax.grid()
    ax.legend(fontsize = lsize - 2)
    
    plt.show()



data = file_read_column("prof_304A.dat", 1)
time = file_read_column("prof_304A.dat", 0)

alpha = np.float(-0.9)

size_const_cond = np.float((0.5)**(-5/2) - 1)
size_const_rad = np.float( (1 - (0.5)**(alpha))/alpha )

temp1 = (1 + time * size_const_cond)**(-2/5)
temp_rad = (1 - alpha * time * size_const_rad )**(1/alpha)


general_plot(time, temp1, temp_rad, data)