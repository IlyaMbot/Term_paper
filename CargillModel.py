import matplotlib.pyplot as plt
import numpy as np

#initial constants
T0 = 1
end = 50

#loop's properties 
loop_lenght = np.float(5* 10**8)
conc = np.float(10**10)
temp_max = np.float(1.5 * 10**7)

alpha = np.float(-0.5)

#timescales
timescale_cond = 1
#np.float(4*10**(-10) * conc * loop_lenght**2 / temp_max**(2.5))
timescale_rad = 10
#np.float(3.45 * 10**(3) * temp_max**(1.5) / conc)

print(timescale_rad, "\n", timescale_cond)

def general_plot(time, data1, data2):
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(time, data1)
    ax.plot(time, data2)
    
    #y = np.arange(0, 1.1, 0.1)

    #ax.axis([-0.1, 51, -0.1, 1.1])

    #plt.yticks(y)
    plt.tight_layout()
    plt.show()


time = np.arange(start = 0, stop = end , step = end/1000)

temp1 = T0 * (1 + time / timescale_cond)**(-2/5)
#temp2 = T0 * (1 + time / timescale_cond )**(-2/7)
temp_rad = T0 * (1 + ((1 - alpha) * time) / (timescale_rad))**(-1/(1 - alpha))


general_plot(time, temp1, temp_rad)
