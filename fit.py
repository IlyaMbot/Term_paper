#! /usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit as cv


def file_read_column(path, number):

    df = pd.read_table(path, header = None, sep = r"\s+")
    return(df.iloc[:,number])


def general_plot(time1, time2, data1, data2):
    lsize = 30
    
    fig, ax = plt.subplots(1, 1)
    
    plt.tight_layout()

    ax.plot(time1, data1, label = "Ashwanden model", linewidth = 3.0)
    ax.plot(time2, data2, label = "Solar flare 304A SDO/AIA", linewidth = 3.0)
    #ax.plot(time2, data2, "bo", label = "Solar flare 304A SDO/AIA")

    
    y = np.arange(0, 1.1, 0.1)
    x = np.arange(-5, 15, 1)

    
    plt.yticks(y)
    plt.xticks(x)

    ax.set_title("Temperature profile", size = lsize)
    ax.set_xlabel('Time', size = lsize)
    

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize - 5)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize - 5)

    ax.axis([-5, 14.5, -0.1, 1.1])

    ax.tick_params(axis = 'x', rotation = 45)
    ax.grid()
    ax.legend(fontsize = lsize - 2)
    
    plt.show()



alpha = np.float(-0.9)

data = file_read_column("prof_304A.dat", 1)
time2 = file_read_column("prof_304A.dat", 0)

def exp_mod(t, sigma):    
    return(np.exp(( -(t)**2 )/ sigma))




model = []
#sigma = 0.2
#alpha = 18

time1 = np.arange(-6, 15, 0.01)

position = np.argwhere(data  >= 0.5)

pos_tp = int(position[-1])


parm1, parm2 = cv(exp_mod, time2[0:pos_tp], data[0:pos_tp])
print(parm1, parm2)

tp = parm1



def parab_mod(t, alpha):
    global tp
    return( 0.5 * (1 - (t - tp)/alpha)**(2))


parm3, parm4 = cv(parab_mod, time2[pos_tp:-1], data[pos_tp:-1])
print(parm3, parm4)


def Ash_mod(t, sigma, alpha):
    T = np.exp(( -(t)**2 )/ sigma)
    tp = np.sqrt(sigma * np.log(2))
    if( T <= 0.5 and t >= 0):
        return( 0.5 * (1 - (t - tp)/alpha)**(2))
    return(T)

for i in time1:
    model.append( Ash_mod(i, parm1, parm3))


general_plot(time1, time2, model, data)
