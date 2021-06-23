#! /usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def file_read_column(path, number):

    df = pd.read_table(path, header = None, names = ["time", "data", "data2"], sep = r"\s+" )
    df = df.sort_values(by = "time")
    data = df.iloc[:,number]
    return(data.to_numpy())

def general_plot(time, data, data2):
    lsize = 16
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)    

    ax.plot(time, data2, label = "И")
    ax.plot(time, data, label = "Т")
    
    ax.set_title("Cтандартное отклонение", size = lsize * 1.5)
    ax.set_xlabel('Уровень перехода', size = lsize *1.2 )
    ax.set_ylabel('Стандартное отклонение', size = lsize )
    

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize * 0.8 )
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize *0.8 )


    #ax.grid()
    ax.legend(fontsize = lsize )
    
    #plt.tight_layout()
    

    #plt.savefig('/home/conpucter/Desktop/Term_paper/data/result/{}-{}.png'.format(index, pic_ind), transparent=False, dpi=500, bbox_inches="tight")
    plt.show()

def plot_one(time, data):
    lsize = 16
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)  

    min_pos = np.argmin(data)

    ax.plot(time, data, label = "Cтандартное отклонение", linewidth = 5.0)
    ax.plot(time[min_pos], data[min_pos], "rx", label = "Минимальное значение", markersize = 15)

    ax.annotate("Температура перехода = {:.2f}".format(time[min_pos]) + "$T_{max}$" + "\nСтандартное отклонение = {:.3f}".format(data[min_pos]),xy = (time[min_pos], data[min_pos]), xytext = (0.4, 0.15), size = lsize, arrowprops=dict(facecolor='black', shrink=0.05))
    
    ax.set_xlabel('Уровень перехода', size = lsize *1.2 )
    ax.set_ylabel('Стандартное отклонение', size = lsize )
    

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize * 0.8 )
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize *0.8 )


    #ax.grid()
    ax.legend(fontsize = lsize )
    
    #plt.tight_layout()
    

    plt.savefig('/home/conpucter/Desktop/Term_paper/data/result/average_Prof_304(D).png', transparent=False, dpi=500, bbox_inches="tight")
    plt.show()

filename = "lim.txt"

limit = file_read_column(filename, 0)
param1 = file_read_column(filename, 1)
#param2 = file_read_column(filename, 2)

plot_one(limit, param1)
#general_plot(limit, param1, param2)