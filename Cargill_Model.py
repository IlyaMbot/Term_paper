#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np


def general_plot(time, data1, data2):
    lsize = 30
    
    fig, ax = plt.subplots(1, 1)
    
    plt.tight_layout()

    ax.plot(time, data1, label = "Conductive cooling", linewidth = 3.0)
    ax.plot(time, data2, label = "Radiative cooling", linewidth = 3.0)
    
    y = np.arange(0, 1.1, 0.1)
    x = np.arange(0, 5.5, 0.5)

    
    plt.yticks(y)
    plt.xticks(x)

    ax.set_title("Temperature profile", size = lsize)
    ax.set_xlabel('Time', size = lsize)
    

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(lsize - 5)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(lsize - 5)

    ax.axis([-0.1, 2.5, -0.1, 1.1])

    ax.tick_params(axis = 'x', rotation = 45)
    ax.grid()
    ax.legend(fontsize = lsize - 2)
    
    plt.show()

#initial constants
end = 5

#loop's properties 
loop_lenght = np.float(5* 10**8)
conc = np.float(10**10)
temp_max = np.float(1.5 * 10**7)

alpha = np.float(-1.5)

time = np.arange(start = 0, stop = end , step = end/1000)

'''
#timescales
timescale_cond = np.float(4*10**(-10) * conc * loop_lenght**2 / temp_max**(2.5))
timescale_rad = np.float(3.45 * 10**(3) * temp_max**(1.5) / conc) /20042
print(timescale_rad, "\n", timescale_cond)


temp1 = T0 * (1 + time / timescale_cond)**(-2/5)
#temp2 = T0 * (1 + time / timescale_cond )**(-2/7)
temp_rad = T0 * (1 - ((1 - alpha) * time) / (timescale_rad))**(1/(1 - alpha))
'''

size_const_cond = np.float((0.5)**(-5/2) - 1)
size_const_rad = np.float( (1 - (0.5)**(alpha))/alpha )

temp1 = (1 + time * size_const_cond)**(-2/5)
temp_rad = (1 - alpha * time * size_const_rad )**(1/alpha)


general_plot(time, temp1, temp_rad)
