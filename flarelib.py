#! /usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit as cv
import os, glob

def get_file():
    return(0)

#--------------------------------------------------------------------
# Getting data functions
 
def file_read_column(path, col):
    '''
    Reads text file that has table format
    takes path and number of column
    returns column data in numpy.array(?) format
    '''
    df = pd.read_table(path, header = None, names = ["time", "data"], sep = r"\s+" )
    df = df.sort_values(by = "time")
    data = df.iloc[:, col]
    return(data.to_numpy())

def get_sav_data(file_name, name):
    readed = scio.readsav(file_name, idict=None, python_dict=False, uncompressed_file_name=None, verbose=False)
    return(readed['{}'.format(name)])

#--------------------------------------------------------------------
#plotting functions

def general_plot(time, time2, data, data2, lim, index, aver, lim_time):
    #TODO make in *kwargs(?)
    begin_time = 0
    end_time = 13
    '''
    TODO: add *kwargs and rename the vars
    Plotting function takes time and data and plots graphs
    '''
    lsize = 20
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)    

    ax.plot(time2, data2, label = "Профиль №{}".format(index+1), linewidth = 5.0, color = "black")
    ax.plot(time, data, label = "Результат фитирования", linewidth = 3.0, color = "red")
    
    # TODO: refact y and x with max val of data & time?
    y = np.arange(0, 1.1, 0.1)
    x = np.arange(-5, 9, 2)
    
    plt.yticks(y)
    plt.xticks(x)

    #TODO: find the way to print all below optionally (depends on s)
    ax.set_xlabel('Время, [$t_{\\frac{1}{2}}$]', size = lsize *1.2 )
    ax.set_ylabel('Нормированный поток излучения', size = lsize )
    ax.annotate(
        "Температура перехода = {:.2f}".format(lim) + "$T_{max}$" + "\nВремя перехода = {:.2f}".format(lim_time) +
        "\nСтандартное отклонение = {:.3f}".format(aver), xy = (lim_time, lim), xytext = (1.5, 0.5), size = lsize,
        arrowprops = dict(facecolor = 'black', shrink = 0.05))
    
    ax.axis([begin_time, end_time, -0.01, 1.01])

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize )
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize *0.8 )


    ax.tick_params(axis = 'x', rotation = 45)
    ax.legend(fontsize = lsize )
    
    #TODO: make below optional 
    #plt.savefig('/home/conpucter/Desktop/Term_paper/data/result/floatpoint-{}.png'.format(index), transparent=False, dpi=500, bbox_inches="tight")
    plt.show()

#--------------------------------------------------------------------
# Operations with data functions

def normalize_d(data):
    '''
    Data normalization function
    Puts all values in range from 0(min) to 1(max)
    '''
    dmax = np.max(data)
    dmin = data[0]
    data = (data - dmin)/(dmax - dmin)
    return(data)

def normalize_t(data, time):
    '''
    Time normalization function
    Normalize all values 
    '''
    max_pos = np.argwhere(data == np.max(data))
    position = np.argwhere(data  >= 0.5)

    time0 = time[int(max_pos[-1])]
    time = (time - time0) /( time[int(position[-1])] - time0 )
    
    return(time)

#--------------------------------------------------------------------
# Models functions

def A_exp(t, sigma):
    '''
    Impulse phase math function from Aschwanden
    '''
    return(np.exp(( -(t)**2 )/ sigma))

def A_parab(t, alpha):
    '''
    Radiative phase math function from Aschwanden
    '''
    #TODO: remove global vars
    global sig
    global lim
    tp = np.sqrt(sig * np.log(1/lim))
    return( lim * (1 - (t - tp)/alpha)**(2) )


def A_full(t, sigma, alpha, lim):
    '''
    All phases math function from Aschwanden
    '''

    T = np.exp(( -(t)**2 )/ sigma)
    
    if( T <= lim):
        tp = np.sqrt(sigma * np.log(1/lim))
        return(np.float( lim * (1 - (t - tp)/alpha)**(2)))
   
    return(np.float( T ))


def C_cond(t, size):
    return((1 + t * size)**(-2/5))

def C_rad(t, alpha, size):
    return((1 - alpha * t * size )**(-1/ alpha))

def C_full(t, alpha, size_r, size_c, lim):

    T = (1 + t * size_c)**(-2/5)
    
    if(T < lim):
        return((1 - alpha * t * size_r )**(-1/ alpha))
    
    return(T)

#--------------------------------------------------------------------
# 

def gen_cool_t_search(t, size_c, lim):

    T = (1 + t * size_c)**(-2/5)
    
    if(T < lim):
        return(t)

