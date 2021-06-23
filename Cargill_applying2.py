#! /usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit as cv
import scipy.io as scio
import os, glob


def file_read_column(path, number):

    df = pd.read_table(path, header = None, names = ["time", "data"], sep = r"\s+" )
    df = df.sort_values(by = "time")
    data = df.iloc[:,number]
    return(data.to_numpy())


def general_plot(time, time2, data, data2, lim, aver, lim_time, index, alpha, C, R):
    lsize = 20
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)  

    #profile = 1600 + index *100
    #profile = 304  
    profile = "D"
    #profile = "Asch_{}".format(index)
    ax.plot(time2, data2, label = "Профиль (белый свет)", linewidth = 5.0, color = "black")
    #ax.plot(time2, data2, label = "Профиль {} Å".format(profile), linewidth = 5.0, color = "black")
    #ax.plot(time2, data2, label = "Профиль №{}".format(index), linewidth = 5.0, color = "black")
    ax.plot(time, data, label = "Результат фитирования", linewidth = 3.0, color = "red")

    y = np.arange(0, 1.1, 0.1)
    x = np.arange(-5, 17, 2)
    
    plt.yticks(y)
    plt.xticks(x)

    #ax.set_title("Временной профиль (температура)", size = lsize * 1.5)
    ax.set_xlabel('Время, [$t_{\\frac{1}{2}}$]', size = lsize *1.2 )
    ax.set_ylabel('Нормированный поток излучения', size = lsize )
    ax.annotate(
        "Температура перехода = {:.2f}".format(lim) + "$T_{max}$" + "\nВремя перехода = {:.2f}".format(lim_time) +
        "\nСтандартное отклонение = {:.3f}".format(aver), xy = (lim_time, lim), xytext = (4, 0.5), size = lsize,
        arrowprops = dict(facecolor = 'black', shrink = 0.05))
    ax.annotate(
        "$\\alpha$ = {:.2f}, ".format(alpha) + "$\\tau_{r}$" + " = {:.2f}, ".format(R) + "$\\tau_{c}$"
        + " = {:.2f}".format(C), xy = (lim_time, lim), xytext = (4, 0.4), size = lsize)
    

    ax.axis([0, 15, -0.01, 1.01])

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize )
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize *0.8 )

    ax.legend(fontsize = lsize )
    
    #plt.tight_layout()
    #plt.savefig('/home/conpucter/Desktop/Term_paper/data/result/C_result_profile{}.png'.format(profile), transparent=False, dpi=500, bbox_inches="tight")
    plt.show()
    print(index)
    
def get_sav_data(file_name, name):
    readed = scio.readsav(file_name, idict=None, python_dict=False, uncompressed_file_name=None, verbose=False)
    return(readed['{}'.format(name)])


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


def cond_cool(t, size):
    return((1 + t * size)**(-2/5))

def rad_cool(t, alpha, size):
    return((1 + alpha * t * size )**(-1/ alpha))

def gen_cool(t, alpha, size_r, size_c, lim):
    T = (1 + t * size_c)**(-2/5)
    
    if(T < lim):
        return((1 + alpha * t * size_r )**(-1/ alpha))
    
    return(T)


def gen_cool_t_search(t, size_c, lim):

    T = (1 + t * size_c)**(-2/5)
    
    if(T < lim):
        return(t)



#f1 = "data/*.txt"
f2 = "prof*.dat"
f3 = "Dav*.dat"
f4 = "res_data*"
#f5 = "/home/conpucter/GitHub/Term_paper/data/Aschw_data/digtext*.dat"

filenames = glob.glob(f3)
filenames = sorted(filenames, key=os.path.basename)


index = 0
for filename in filenames:

    data = file_read_column(filename, 1)
    time = file_read_column(filename, 0)

    #time = get_sav_data(filename, "time_net1")
    #data = get_sav_data(filename, "int")
        
    maximum = int(np.argwhere(data == np.amax(data))[-1])

    data = data[maximum: -1]
    time = time[maximum: -1]

    data = normalize_d(data)
    time = normalize_t(data, time)
    
    limits = np.arange(0.2 , 1, 0.01)

    avers = []
    sizes = [[],[]]
    alphas = []
    limas = [[],[]]

    for lim in limits:
        #get cond_cool and rad_cool
        position = np.argwhere(data >= lim)
        pos_zero = np.argwhere(data >= 0)

        pos_tp = int(position[-1]) + 1
        
        parm1, parm2 = cv(cond_cool, time[0 : pos_tp], data[0 : pos_tp],bounds = (0, 20))
        parm3, parm4 = cv(rad_cool, time[pos_tp : -1], data[pos_tp : -1], bounds = (0, 20))
        
        size_c = parm1
        size_r = parm3[1]
        alpha = parm3[0]
        sizes[0].append(size_c)
        sizes[1].append(size_r)
        alphas.append(alpha)
        
        time2 = np.arange(0, 15 , 0.01)
        for lim_timea in time2[2:-1]:                
            if cond_cool(lim_timea, size_c) >= rad_cool(lim_timea, alpha, size_r):
                lima = cond_cool(lim_timea, size_c)
                limas[0].append(lim_timea)
                limas[1].append(lima)
                break 

        aver = 0.0        
           
        ind = 0
        for i in time:
            aver += (gen_cool(i, alpha, size_r, size_c, lima) - data[ind])**2
            ind += 1 
            
        #print(aver)
        aver = np.float(np.sqrt(aver / len(time)))
        avers.append(aver)
        
    
    alim = np.argmin(avers)
    size_c = sizes[0][alim][0]
    size_r = sizes[1][alim]
    lim_time = limas[0][alim]
    lim = limas[1][alim][0]
    aver = avers[alim]
    alpha = alphas[alim]

    model = []
    time2 = np.arange(0, 15 , 0.01)
           
    
    for i in time2:
        model.append( gen_cool(i, alpha, size_r, size_c, lim))

    print("Rad = {}, Cond = {}".format( 1/size_r, 1/size_c))

   
    
    general_plot(time2, time, model, data, lim, aver, lim_time, index, alpha, (1/size_c), (1/size_r) )
    index +=1