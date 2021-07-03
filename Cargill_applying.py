#! /usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit as cv
import scipy.io as scio
import os, glob



def general_plot(time, time2, data, data2, lim, aver, lim_time, index):
    lsize = 16
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)  

    profile = 1600 + index *100  

    #ax.plot(time2, data2, label = "Профиль (белый свет)", linewidth = 5.0, color = "black")
    ax.plot(time2, data2, label = "Профиль ({})".format(profile), linewidth = 5.0, color = "black")
    ax.plot(time, data, label = "Результат фитирования", linewidth = 3.0, color = "red")

    y = np.arange(0, 1.1, 0.1)
    x = np.arange(-5, 17, 2)
    
    plt.yticks(y)
    plt.xticks(x)

    ax.set_title("Временной профиль (температура)", size = lsize * 1.5)
    ax.set_xlabel('Время, [$t_{\\frac{1}{2}}$]', size = lsize *1.2 )
    ax.set_ylabel('Температура, (нормированные значения)', size = lsize )
    ax.annotate(
        "Температура перехода = {:.2f}".format(lim) + "$T_{max}$" + "\nВремя перехода = {:.2f}".format(lim_time) +
        "\nСтандартное отклонение = {:.3f}".format(aver), xy = (lim_time, lim), xytext = (6, 0.5), size = lsize,
        arrowprops = dict(facecolor = 'black', shrink = 0.05))
    

    ax.axis([0, 15, -0.01, 1.01])

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize * 0.8 )
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize *0.8 )


    ax.tick_params(axis = 'x', rotation = 45)
    #ax.grid()
    ax.legend(fontsize = lsize )
    
    #plt.tight_layout()
    #plt.savefig('/home/conpucter/Desktop/Term_paper/data/result/result_profile{}.png'.format(profile), transparent=False, dpi=500, bbox_inches="tight")
    plt.show()
    print(index)



f1 = "data/*.txt"
f2 = "prof*.dat"
f3 = "Dav*.dat"
f4 = "res_data*"
f5 = "/home/conpucter/GitHub/Term_paper/data/Aschw_data/digtext*.dat"

filenames = glob.glob(f2)
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

    for lim in limits:

        position = np.argwhere(data >= lim)
        pos_zero = np.argwhere(data >= 0)

        pos_tp = int(position[-1]) + 1
        
        parm1, parm2 = cv(cond_cool, time[0 : pos_tp], data[0 : pos_tp])
        parm3, parm4 = cv(rad_cool, time[pos_tp : -1], data[pos_tp : -1], bounds = ([0, -20], 20))
        
        size_c = parm1
        size_r = parm3[1]
        alpha = parm3[0]
        sizes[0].append(size_c)
        sizes[1].append(size_r)
        alphas.append(alpha)
        
        aver = 0.0        
           
        ind = 0
        for i in time:
            aver += (gen_cool(i, alpha, size_r, size_c, lim) - data[ind])**2
            ind += 1 
            
        print(aver)
        aver = np.float(np.sqrt(aver / len(time)))
        avers.append(aver)
        
    
    alim = np.argmin(avers)
    size_c = sizes[0][alim][0]
    size_r = sizes[1][alim]
    model = []
    time2 = np.arange(0, 15 , 0.01)

    for lim_time in time2[2:-1]:                
        if cond_cool(lim_time, sizes[0][alim][0]) >= rad_cool(lim_time, alphas[alim], sizes[1][alim]):
            lim = cond_cool(lim_time, sizes[0][alim][0])
            print(lim, lim_time)

            break 

    '''        
    for i in time2:                
            lim_time = gen_cool_t_search(i, size_c, limits[alim])
            if lim_time != None:
                break 
    '''
    
    for i in time2:
        model.append( gen_cool(i, alphas[alim], size_r, size_c, lim))
    #print("cond =", sizes[0][alim][0], "\nrad =", sizes[1][alim], "\nalpha = ", alphas[alim])
    print(size_r, size_c)

   
    
    general_plot(time2, time, model, data, lim, avers[alim], lim_time, index)
    '''
    for i in range(len(limits)):
        print( limits[i], avers[i] )
    '''    
    index +=1