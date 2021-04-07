#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

#constants
k_boltzmann = np.float(1.38 * 10**(-16))
mass_proton = np.float(1.67 * (10)**(-24))
mu = np.float(1.27)
lambda0 = np.float(10**(-17.73))
n_p = np.float(1.2)
n_cool = 4
k = np.float(9.2 * (10)**(-7))

L = np.float(27.5 * 10**6)
Sh_fp = 1
Sh_apx = -1
#______________________________________________________________________________

#heating non-universal factor
qh = (L**2)/(1-(1+L/Sh_fp)* np.exp(-L/Sh_fp))
#______________________________________________________________________________

tau_heat = 210 # dispertion


#time:
ts = 0
te = 5000
tstep = 1
tm = 882 # time of temp max

tp = int((7 * np.log(2))**(0.5) * tau_heat + tm)

tau_cool = 1480  #exper


t_heat = np.arange(ts, tp, tstep) 
t_gen = np.arange(ts, te, tstep)
t_cool = np.arange(tp, tp + tau_cool, tstep)

n_cool = 1
#______________________________________________________________________________


H_max = 7.5

H0f = H_max  # VHR at the footpoint
H0a = H_max * np.exp(-3)

# temperature of loop

Tmax = (7 * L**2 * H0f)/(4 * k * qh)
#Tmax = 10**7
Tp = Tmax/2

# T(x=0) footpoint 
T0 = 0 
 

#t_cool1 = np.arange(tp, tmin, tstep)
#t_cool2 = np.arange(tmin, te, tstep)



# Temperature 
T_time = np.zeros(len(t_gen))
T_time[0 : int((tp-ts)/tstep)] = Tmax * np.exp((-1)*( ( t_heat - tm )**2 )/(7 * (tau_heat)**2 )) 
T_time[int((tp-ts)/tstep) : (tp + tau_cool) ] = Tp * ((1 - ((t_cool-tp)/(tau_cool*n_cool)))**(2))
T_time[(tp + tau_cool): len(t_gen)] = T0

#VHR itself:
Eh = H0f * np.exp((-1)*((t_gen-tm)**2)/(2*tau_heat**2))
#______________________________________________________________________________


#ploting...
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Temperature profile")

ax.set_xlabel('Time, s')
ax.set_ylabel('Temperature, K')

ax.axis([0, 5000 , 0, 40 * 10**6])
ax.plot(t_gen, T_time, linewidth = 3.0)
ax.plot([tp, tp + 1],[40 * 10**6 ,0], ls = "--", color = 'blue')
ax.plot([tm, tm + 1],[40 * 10**6 ,0], ls = "--", color = 'red')
ax.plot([tp + tau_cool , tp + tau_cool + 1],[40 * 10**6 ,0], ls = "--", color = 'black')

ax.grid()
plt.show()
#______________________________________________________________________________
