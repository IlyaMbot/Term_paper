#! /usr/bin/env python3
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np

def plot_one(time, data):
    lsize = 16
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)  

    ax.plot(time, data, label = "1600", linewidth = 5.0)

    ax.set_xlabel('time', size = lsize *1.2 )
    ax.set_ylabel('flux', size = lsize )
    

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize * 0.8 )
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize *0.8 )


    #ax.grid()
    ax.legend(fontsize = lsize )
    
    #plt.tight_layout()
    

    #plt.savefig('/home/conpucter/Desktop/Term_paper/data/result/average_Prof_304(D).png', transparent=False, dpi=500, bbox_inches="tight")
    plt.show()

file_name = "res_data_1700.sav"
readed = scio.readsav(file_name, idict=None, python_dict=False, uncompressed_file_name=None, verbose=False)
time = readed['time_net1']
data = readed['int']

plot_one(time, data)