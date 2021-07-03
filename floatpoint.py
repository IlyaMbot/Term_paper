#! /usr/bin/env python3

from scipy.optimize import curve_fit as cv
import numpy as np
import os, glob
import flarelib as fl
    

f2 = "/home/conpucter/GitHub/Term_paper/data/Aschw_data/digtext*.dat"
f1 = "/home/conpucter/GitHub/Term_paper/data/1.txt"

filenames = glob.glob(f2)
filenames = sorted(filenames, key=os.path.basename)

index = 0
for filename in filenames:

    data = fl.file_read_column(filename, 1)
    time = fl.file_read_column(filename, 0)
        
    maximum = int(np.argwhere(data == np.amax(data))[-1])

    data = fl.normalize_d(data)
    time = fl.normalize_t(data, time)
    
    data = data[maximum: -1]
    time = time[maximum: -1]
    
    limits = np.arange(0.1, 0.99, 0.01)
    avers = []
    parms = [[],[]]

    for lim in limits:
        position = np.argwhere(data >= lim)
        pos_zero = np.argwhere(data >= 0)

        pos_tp = int(position[-1]) + 1
        pos_zero = int(pos_zero[-1])

        parm1, parm2 = cv(fl.A_exp, time[0 : pos_tp], data[0 : pos_tp])
        sig = parm1[0]
        parm3, parm4 = cv(fl.A_parab, time[pos_tp : pos_zero], data[pos_tp : pos_zero])

        alpha = parm3[0]
        parms[0].append(sig)
        parms[1].append(alpha)
        
        aver = 0.0 
        ind = 0
        for i in time:
            aver += (fl.A_full(i, parm1[0], parm3[0], lim) - data[ind])**2
            ind += 1 
            
        aver = np.float(np.sqrt(aver / len(time)))
        avers.append(aver)
        
    
    alim = np.argmin(avers)
                   
    time2 = np.arange(0, 15 , 0.01)
    model = []

    for lim_time in time2:  
        if( fl.A_exp(lim_time, parms[0][alim]) <= limits[alim]):
            break 
    
    te = lim_time + parms[1][alim]

    for i in time2:
        if (i < te):
            model.append( fl.A_full(i, parms[0][alim], parms[1][alim], limits[alim]))
        else:
            model.append(0)    
    

    fl.general_plot(time2, time, model, data, limits[alim], index, avers[alim], lim_time)

    index +=1