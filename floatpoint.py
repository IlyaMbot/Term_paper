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


def general_plot(time, time2, data, data2, lim, index, aver, lim_time):
    lsize = 20
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)    

    ax.plot(time2, data2, label = "Профиль №{}".format(index+1), linewidth = 5.0, color = "black")
    ax.plot(time, data, label = "Результат фитирования", linewidth = 3.0, color = "red")

    y = np.arange(0, 1.1, 0.1)
    x = np.arange(-5, 9, 2)

    
    plt.yticks(y)
    plt.xticks(x)

    ax.set_xlabel('Время, [$t_{\\frac{1}{2}}$]', size = lsize *1.2 )
    ax.set_ylabel('Нормированный поток излучения', size = lsize )
    ax.annotate(
        "Температура перехода = {:.2f}".format(lim) + "$T_{max}$" + "\nВремя перехода = {:.2f}".format(lim_time) +
        "\nСтандартное отклонение = {:.3f}".format(aver), xy = (lim_time, lim), xytext = (1.5, 0.5), size = lsize,
        arrowprops = dict(facecolor = 'black', shrink = 0.05))
    

    ax.axis([0, 7, -0.01, 1.01])

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize )
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize *0.8 )


    ax.tick_params(axis = 'x', rotation = 45)
    ax.legend(fontsize = lsize )
    
    #plt.tight_layout()
    
    #plt.savefig('/home/conpucter/Desktop/Term_paper/data/result/floatpoint-{}.png'.format(index), transparent=False, dpi=500, bbox_inches="tight")
    plt.show()
    print(index)
    

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



def exp_mod(t, sigma):    
    return( np.exp(( -(t)**2 )/ sigma) )

def parab_mod(t, alpha):
    global sig
    global lim
    tp = np.sqrt(sig * np.log(1/lim))
    return( lim * (1 - (t - tp)/alpha)**(2) )


def Ash_mod(t, sigma, alpha, lim):

    T = np.exp(( -(t)**2 )/ sigma)
    
    if( T <= lim):
        tp = np.sqrt(sigma * np.log(1/lim))
        return(np.float( lim * (1 - (t - tp)/alpha)**(2)))
   
    return(np.float( T ))


f2 = "/home/conpucter/GitHub/Term_paper/data/Aschw_data/digtext*.dat"
f1 = "/home/conpucter/GitHub/Term_paper/data/1.txt"

filenames = glob.glob(f2)
filenames = sorted(filenames, key=os.path.basename)

index = 0
for filename in filenames:

    data = file_read_column(filename, 1)
    time = file_read_column(filename, 0)
        
    maximum = int(np.argwhere(data == np.amax(data))[-1])

    data = normalize_d(data)
    time = normalize_t(data, time)
    
    data = data[maximum: -1]
    time = time[maximum: -1]
    
    #print(time)
    #limits = np.arange(0.1 , 0.99, 0.01)
    limits = np.arange(0.1, 0.99, 0.01)
    avers = []
    parms = [[],[]]

    for lim in limits:
        position = np.argwhere(data >= lim)
        pos_zero = np.argwhere(data >= 0)

        pos_tp = int(position[-1]) + 1
        pos_zero = int(pos_zero[-1])

        parm1, parm2 = cv(exp_mod, time[0 : pos_tp], data[0 : pos_tp])
        sig = parm1[0]
        parm3, parm4 = cv(parab_mod, time[pos_tp : pos_zero], data[pos_tp : pos_zero])

        alpha = parm3[0]
        parms[0].append(sig)
        parms[1].append(alpha)
        
        aver = 0.0 
        ind = 0
        for i in time:
            aver += (Ash_mod(i, parm1[0], parm3[0], lim) - data[ind])**2
            ind += 1 
            
        #print(aver)
        aver = np.float(np.sqrt(aver / len(time)))
        #print(aver)
        avers.append(aver)
        
    
    alim = np.argmin(avers)
    #print(parms)
                   
    time2 = np.arange(0, 15 , 0.01)
    model = []

    for lim_time in time2:  
        if( exp_mod(lim_time, parms[0][alim]) <= limits[alim]):
            break 
    
    te = lim_time + parms[1][alim]
    #te = parms[0][alim] + parms[1][alim]

    for i in time2:
        if (i < te):
            model.append( Ash_mod(i, parms[0][alim], parms[1][alim], limits[alim]))
        else:
            model.append(0)    
    
    

    general_plot(time2, time, model, data, limits[alim], index, avers[alim], lim_time)

    index +=1