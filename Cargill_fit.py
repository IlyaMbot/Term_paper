#! /usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit as cv
import os, glob


def file_read_column(path, number):

    df = pd.read_table(path, header = None, names = ["time", "data"], sep = r"\s+" )
    df = df.sort_values(by = "time")
    data = df.iloc[:,number]
    return(data.to_numpy())


def general_plot(time, time2, data, data1, data2, pic_ind):
    lsize = 16
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)    

    ax.plot(time2, data2, label = "Модельный профиль", linewidth = 5.0, color = "black")
    ax.plot(time, data, label = "Результат фитирования (излучение)", linewidth = 3.0, color = "red")
    ax.plot(time, data1, label = "Результат фитирования (теплопроводность)", linewidth = 3.0, color = "yellow")
    

    y = np.arange(0, 1.1, 0.1)
    x = np.arange(-5, 10, 2)

    
    plt.yticks(y)
    plt.xticks(x)

    '''
    ax.set_title("Temperature profile", size = lsize)
    ax.set_xlabel('Time', size = lsize)
    ax.set_ylabel('Relative flux', size = lsize)
    '''
    ax.set_title("Временной профиль (температура)", size = lsize * 1.5)
    ax.set_xlabel('Время, [$t_{\\frac{1}{2}}$]', size = lsize *1.2 )
    ax.set_ylabel('Температура, (нормированные значения)', size = lsize )
    

    ax.axis([0, 11, -0.01, 1.01])

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize * 0.8 )
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize *0.8 )


    ax.tick_params(axis = 'x', rotation = 45)
    #ax.grid()
    ax.legend(fontsize = lsize )
    
    #plt.tight_layout()
    #plt.show()

    plt.savefig('/home/conpucter/Desktop/Term_paper/data/result/{}.png'.format(pic_ind), transparent=False, dpi=500, bbox_inches="tight")
    print(pic_ind)
    

def normalize_d(data):
    dmax = np.max(data)
    dmin = data[0]
    data = (data - dmin)/(dmax - dmin)
    return(data)


def normalize_t(data, time):
    max_pos = np.argwhere(data == np.max(data))
    position = np.argwhere(data  >= 0.5)

    time0 = time[int(max_pos[-1])]
    time = (time - time0) /( time[int(position[-1])] - time0 )
    
    return(time)


def cond_cool(t, size):
    #size = np.float((0.5)**(-5/2) - 1)
    temp = (1 + t * size)**(-2/5)
    return(temp)

def rad_cool(t, alpha, size):
    #alpha = np.float(-1.5)
    #size = np.float( (1 - (0.5)**(-1.5))/ (-1.5) )
    temp_rad = (1 - alpha * t * size )**(-1/ alpha)
    return(temp_rad)





filenames = glob.glob("data/*.txt")
filenames = sorted(filenames, key=os.path.basename)

pic_ind = 0

for filename in filenames:

    data = file_read_column(filename, 1)
    time = file_read_column(filename, 0)

    data = normalize_d(data)
    time = normalize_t(data, time)

    time_beginning = np.argwhere(data == np.max(data))
    time_beginning = int(time_beginning[-1])

    data0 = data[time_beginning : -1]
    time0 = time[time_beginning : -1]

    pos_zero = np.argwhere(data >= 0)
    pos_zero = int(pos_zero[-1])
    
    data = data0[0 : pos_zero]
    time = time0[0 : pos_zero]
   
    time2 = np.arange(0, 15 , 0.1)


    model1 = []
    model2 = []

    parm1, parm2 = cv(rad_cool, time, data, bounds = ([0, -20], 20))
    print(parm1, parm2)

    parm3, parm4 = cv(cond_cool, time, data)
    print(parm3, parm4)

    for i in time2:
        model1.append( rad_cool(i, parm1[0], parm1[1]) )
        model2.append( cond_cool(i, parm3) )

       

    general_plot(time2, time0, model1, model2, data0, pic_ind)

    pic_ind += 1
    