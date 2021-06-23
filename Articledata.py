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


def general_plot(time, time2, data, data2, aver, index):
    lsize = 20
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)  

    
    ax.plot(time2, data2, label = "Профиль №{}".format(index), linewidth = 5.0, color = "black")
    ax.plot(time, data, label = "Результат фитирования", linewidth = 3.0, color = "red")

    y = np.arange(0, 1.1, 0.1)
    x = np.arange(-5, 17, 2)
    
    plt.yticks(y)
    plt.xticks(x)

    #ax.set_title("Временной профиль", size = lsize * 1.5)
    ax.set_xlabel('Время, [$t_{\\frac{1}{2}}$]', size = lsize *1.2 )
    ax.set_ylabel('Нормированный поток излучения', size = lsize )
    ax.annotate(" S = {:.3f}".format(aver[0]), xy = (0, 0), xytext = (3, 0.65), size = lsize*1.5)
    ax.annotate(" $\\frac{\Delta T}{T_{max}}$" + " = {}%".format(int(aver[1] * 100)), xy = (0, 0), xytext = (3, 0.5), size = lsize*1.5)
    

    ax.axis([-4, 7, -0.01, 1.01])

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize )
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize * 0.8)


    ax.legend(fontsize = lsize * 0.8 )
    
    plt.tight_layout()
    #plt.savefig('/home/conpucter/Desktop/Term_paper/data/result/result_profile{}.png'.format(index), transparent=False, dpi=500, bbox_inches="tight")
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
    global size_c
    tp = np.sqrt(size_c * np.log(2))
    return( 0.5 * (1 - (t - tp)/alpha)**(2) )

def Ash_mod(t, sigma, alpha):
    
    T = np.exp(( -(t)**2 )/ sigma)
    tp = np.sqrt(sigma * np.log(2))

    if( T <= 0.5 and t >= 0):
        return(np.float( 0.5 * (1 - (t - tp)/alpha)**(2)))
   
    return(np.float(T))


def aver_sq(data, time):
    aver = 0.0 
    ind = 0
    for i in time:
        aver += (Ash_mod(i, parm1, parm3[0]) - data[ind])**2
        ind += 1 
    aver = np.float(np.sqrt(aver / len(time)))
    return(aver)

def aver_lin(data, time):
    aver = 0.0 
    ind = 0
    for i in time:
        aver += abs(Ash_mod(i, parm1, parm3[0]) - data[ind])
        ind += 1 
    aver = np.float(aver / len(time))
    return(aver)

f1 = "data/Aschw_data/digtext1.dat"
f2 = "data/Aschw_data/digtext*.dat"

filenames = glob.glob(f2)
filenames = sorted(filenames, key=os.path.basename)

pic_ind = 1

for filename in filenames:

    data = file_read_column(filename, 1)
    time = file_read_column(filename, 0)

    data = normalize_d(data)
    time = normalize_t(data, time)
    

    position = np.argwhere(data >= 0.5)
    pos_zero = np.argwhere(data >= 0)

    pos_tp = int(position[-1])
    pos_zero = int(pos_zero[-1])


    parm1, parm2 = cv(exp_mod, time[0:pos_tp], data[0:pos_tp])
    size_c = parm1
    
    parm3, parm4 = cv(parab_mod, time[pos_tp : pos_zero], data[pos_tp : pos_zero])
    
    
    size_r = parm3[0]
    te = size_c + size_r
   
    time2 = np.arange(-4, 10 , 0.01)

    model = []

    for i in time2:
        if (i < te):
            model.append( Ash_mod(i, parm1, parm3[0]))
        else:
            model.append(0)     
           
    aver = [aver_sq(data, time), aver_lin(data, time)]

    general_plot(time2, time, model, data, aver, pic_ind)
    pic_ind += 1
    