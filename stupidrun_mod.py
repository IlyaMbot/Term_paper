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


def general_plot(time, time2, data, data2, pic_ind, index, aver, lim_time):
    lsize = 16
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)    

    ax.plot(time2, data2, label = "Модельный профиль", linewidth = 5.0, color = "black")
    ax.plot(time, data, label = "Результат фитирования", linewidth = 3.0, color = "red")

    y = np.arange(0, 1.1, 0.1)
    x = np.arange(-5, 17, 2)
    
    plt.yticks(y)
    plt.xticks(x)

    ax.set_title("Временной профиль (температура)", size = lsize * 1.5)
    ax.set_xlabel('Время, [$t_{\\frac{1}{2}}$]', size = lsize *1.2 )
    ax.set_ylabel('Температура, (нормированные значения)', size = lsize )
    ax.annotate("Температура перехода = {:.2f}".format(lim) + "$T_{max}$",xy = (4, 0.6), xytext = (6, 0.7), size = lsize)
    ax.annotate("Время перехода = {:.2f}".format(lim_time), xy = (4, 0.6), xytext = (6.5, 0.6), size = lsize)
    ax.annotate("Стандартное отклонение = {:.3f}".format(aver),xy = (4, 0.4), xytext = (5.5, 0.5), size = lsize)
    

    ax.axis([0, 15, -0.01, 1.01])

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize * 0.8 )
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize *0.8 )


    ax.tick_params(axis = 'x', rotation = 45)
    #ax.grid()
    ax.legend(fontsize = lsize )
    
    #plt.tight_layout()
    
    pic_ind = int(float("{:.2f}".format(pic_ind)) *100)
    #plt.savefig('/home/conpucter/Desktop/Term_paper/data/result/{}-{}.png'.format(index, pic_ind), transparent=False, dpi=200, bbox_inches="tight")
    plt.show()
    #print(pic_ind, index)
    

def normalize_d(data):
    dmax = np.max(data)
    dmin = np.min(data)
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
    
    if( T < lim):
        tp = np.sqrt(sigma * np.log(1/lim))
        return(np.float( lim * (1 - (t - tp)/alpha)**(2)))
   
    return(np.float( T ))

def Ash_mod_t_search(t, sigma, alpha, lim):

    T = np.exp(( -(t)**2 )/ sigma)
    
    if( T <= lim):
        return(t)

f1 = "data/*.txt"
f2 = "data/Aschw_data/digtext1.dat"

filenames = glob.glob(f2)
filenames = sorted(filenames, key=os.path.basename)



index = 0
for filename in filenames:

    data = file_read_column(filename, 1)
    time = file_read_column(filename, 0)
        
    maximum = int(np.argwhere(data == np.amax(data))[-1])

    data = data[maximum: -1]
    time = time[maximum: -1]

    data = normalize_d(data)
    time = normalize_t(data, time)
    
    limits = np.arange(0.90 , 0.99, 0.01)

    avers = []
    parms = [[],[]]

    for lim in limits:

        position = np.argwhere(data >= lim)
        pos_zero = np.argwhere(data >= 0)

        pos_tp = int(position[-1]) + 1
        pos_zero = int(pos_zero[-1])

        summ = []
        parm1, parm2 = cv(exp_mod, time[0 : pos_tp], data[0 : pos_tp])
        sig = parm1
        parm3, parm4 = cv(parab_mod, time[pos_tp : pos_zero], data[pos_tp : pos_zero])

        te = parm1 + parm3

        parm2 = np.sqrt(np.diag(parm2))

        aver = [parm2[0], parm2[1]]
        avers.append(parm2[0]*2 + parm2[1])
        parms[0].append(parm1)
        parms[1].append(parm3)

    #position = int(np.argwhere(avers == np.amin(avers)))
               
    sigma = parms[0][position]
    alpha = parms[1][position]
    aver = avers[position]
    te = sigma + alpha 

    time2 = np.arange(0, 15 , 0.01)
    model = []

    for i in time2:
        if (i < te):
            model.append( Ash_mod(i, sigma, alpha, limits[position]))
        else:
            model.append(0)    
       
    
    for i in time2:                
        lim_time = Ash_mod_t_search(i, parm1, parm3[0], lim)
        if lim_time != None:
            print(lim_time)
            break 
        
            

    for i in time:
        if (i < te):
            aver += (Ash_mod(i, parm1, parm3[0], lim) - data[ind])**2
        else:   
            aver += (data[ind])**2 
        ind += 1

        
    aver = np.sqrt(aver / len(time))
    print(parm1, parm3[0])
    general_plot(time2, time, model, data, lim, index, aver, lim_time)
    
    index +=1