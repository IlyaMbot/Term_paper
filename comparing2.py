#! /usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def file_read_column(path, number):

    df = pd.read_table(path, header = None, sep = r"\s+")
    return(df.iloc[:,number])


def general_plot(time1, time2, data1, data2):
    lsize = 30
    
    fig, ax = plt.subplots(1, 1)
    
    plt.tight_layout()

    ax.plot(time1, data1, label = "Ashwanden model", linewidth = 3.0)
    ax.plot(time2, data2, label = "Solar flare 304A SDO/AIA", linewidth = 3.0)
    #ax.plot(time, data3, label = "Solar flare 304A SDO/AIA", linewidth = 3.0)
    
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

def Ash_mod(t, sigma, alpha):
    T = np.exp(( -(t)**2 )/ sigma)
    tp = np.sqrt(sigma * np.log(2))
    if( T <= 0.5 and t >= 0):
        return( 0.5 * (1 - (t - tp)/alpha)**(2))
    return(T)

model = []
sigma = 0.2
alpha = 18
A = 1/(1 - np.sqrt(0.5))

time1 = np.arange(-6, 15, 0.01)

for i in time1:
    model.append( Ash_mod(i, sigma, alpha))


#model[0 : int((tp-ts)/tstep)] = Tmax * np.exp((-1)*( ( t_heat - tm )**2 )/(7 * (tau_heat)**2 )) 
#T_time[int((tp-ts)/tstep) : (tp + tau_cool) ] = Tp * ((1 - ((t_cool-tp)/(tau_cool*n_cool)))**(2))
#T_time[(tp + tau_cool): len(t_gen)] = T0


general_plot(time1, time2, model, data)