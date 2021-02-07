import matplotlib.pyplot as plt
import numpy as np

T0 = 1
end = 1

loop_lenght = 1500
conc = 4000
temp_max = np.float(2)

#timescales
timescale_cond = np.float(4*10**(-10) * conc * loop_lenght / temp_max**(-2.5))


def general_plot(time, data1, data2):
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(time, data1)
    ax.plot(time, data2)
        
    plt.tight_layout()
    plt.show()


time = np.arange(start = 0, stop = end , step = 0.001)

temp1 = T0 * (1 + time / timescale_cond)**(-2/5)
temp2 = T0 * (1 + time / timescale_cond )**(-2/7)


general_plot(time, temp1, temp2)
