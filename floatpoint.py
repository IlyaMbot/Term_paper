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


def general_plot(time, time2, data, data2, pic_ind):
    lsize = 16
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)    

    ax.plot(time2, data2, label = "Модельный профиль", linewidth = 5.0, color = "black")
    ax.plot(time, data, label = "Результат фитирования", linewidth = 3.0, color = "red")

    y = np.arange(0, 1.1, 0.1)
    x = np.arange(-5, 11, 2)

    
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
    plt.show()

    #plt.savefig('/home/conpucter/Desktop/Term_paper/data/result/{}.png'.format(pic_ind), transparent=False, dpi=500, bbox_inches="tight")
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


'''
def exp_mod(t, sigma):    
    return( np.exp(( -(t)**2 )/ sigma) )

def parab_mod(t, alpha):
    global sig
    tp = np.sqrt(sig * np.log(2))
    return( 0.5 * (1 - (t - tp)/alpha)**(2) )
'''

def Ash_mod(t, sigma, alpha):
    
    T = np.exp(( -(t)**2 )/ sigma)

    if( T.any() <= 0.5):
        
        tp = np.sqrt(sigma * np.log(2))
        return(np.float( 0.5 * (1 - (t - tp)/alpha)**(2)))
    return(T)


filenames = glob.glob("data/1.txt")
filenames = sorted(filenames, key=os.path.basename)

pic_ind = 0

for filename in filenames:

    data = file_read_column(filename, 1)
    time = file_read_column(filename, 0)

    data = normalize_d(data)
    time = normalize_t(data, time)
    
    maximum = int(np.argwhere(data == np.amax(data))[-1])
    position = np.argwhere(data >= 0.5)
    pos_zero = np.argwhere(data >= 0)

    pos_zero = int(pos_zero[-1])


    parm1, parm2 = cv(Ash_mod, time[maximum:-1], data[maximum:-1])
    print(parm1, parm2)

    te = parm1[1] + parm1[0]
    
   
    time2 = np.arange(0, 15 , 0.01)
    

    model = []

    for i in time2:
        if (i < te):
            model.append( Ash_mod(i, parm1[0], parm1[1]))
        else:
            model.append(0)    


    general_plot(time2, time, model, data, pic_ind)
    pic_ind += 1