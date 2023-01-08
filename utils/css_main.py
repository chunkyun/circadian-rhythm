import pandas as pd
from pandas import Series, DataFrame
import math
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as img
import os
import sklearn
import datetime
from scipy.integrate import odeint
import scipy
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from utils.data_matching import Data_matching 
from utils.calcul_css import Calcul_CSS
from utils.PCR import PCR
from utils.PCR_shift import PCR_shift



def main(sleep_light_df, waso_df):

    time_interval = 2  
    time_start = 24    
    tau_c = 24.09   

    patt = sleep_light_df.iloc[:,0]
    light = sleep_light_df.iloc[:,1]

    if patt.tail(1).item()=='Sleep':
        patt = patt.append(pd.Series('Wake'),ignore_index=True)
        light = light.append(pd.Series(250), ignore_index=True)

    waso = waso_df.iloc[:,0]
    main_sleep =  np.where(waso_df.iloc[:,1]=='M')                       

    Q_max = 100 ; theta = 10 ; sigma = 3  
    mu = 4.2 ; coef_y = 0.8 ; const = 1 ; v_vh = 1.01 ; coef_x = -0.16                 
    gate = 1       

    start_num = 120 

    if time_start < 12:
        time_start = time_start + 24


    it_first= np.linspace(0, 24*start_num, (24*start_num)*30+1)        
    it2 = it_first.copy()

    for k in range(len(it2)):
        it2[k] =  it2[k] - 24 * (np.floor(it2[k]/24)) 
    # Light on from 6 a.m. to 21:42 a.m. as 250 lux [*]:= i_first                                                                      
    i_first = np.zeros(len(it_first))   
    # Reasonable light on timing according to
    # baseline sleep-wake schedule (sleep occurs between 22:00-6:00) 
    for j in range(len(i_first)):                                                 
        if (it2[j] >= 6) & (it2[j] < 22-0.3):                                       
            i_first[j] = 250 


    tspan_first = it_first.copy()                                                  

    V_00 =  [-0.5, -0.25, 5, 0, -11, 14.25]


    it = it_first ; i=i_first  
    sol = odeint(func=PCR, y0=V_00, t=tspan_first, 
                 args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

    y0 = sol ; t0 = tspan_first

    y_maximum = np.max(y0[-24*30-1:,4])
    max_index = np.where(y0[-24*30-1:,4]==y_maximum)

    y_minimum = np.min(y0[-24*30-1:,4])
    min_index = np.where(y0[-24*30-1:,4]==y_minimum)

    wake_effort_Vm = y_maximum
    wake_effort_Vv = np.min(y0[-24*30+int(max_index[0]),3]) 

    sleep_effort_Vm = y_minimum
    sleep_effort_Vv = np.max(y0[-24*30+int(min_index[0]),3]) 

    # % Track homeostatic sleep pressure and circadian rhythm according to       
    # % sleep-wake pattern and light exposure

    tspan = np.arange(24*start_num+time_start,24*start_num+time_start+(len(patt)-1)*time_interval/60+0.0001 ,time_interval/60)
    it = tspan.copy()

    # % Get sleep time / wake time from data 
    sleep = np.where(patt == 'Sleep')                                              
    real_time = np.where((sleep[0][1:] - sleep[0][:-1]) !=1)                                            
    real_wk_time = sleep[0][real_time[0]]+1
    real_sl_time = sleep[0][real_time[0]+1]

    real_wk_time = real_wk_time.tolist().copy()
    real_wk_time.append(sleep[0][-1]+1)
    real_sl_time = real_sl_time.tolist().copy()
    real_sl_time.insert(0, sleep[0][0])                                 

    real_wk_time_origin = real_wk_time.copy()
    real_sl_time_origin = real_sl_time.copy()

    for j in range(len(real_sl_time)):
        if np.ceil(waso[j]/time_interval) != 0:
            patt[int(real_wk_time[j] - np.ceil(waso[j]/time_interval)) : int(real_wk_time[j])] = 'Wake'
            real_wk_time[j] = real_wk_time[j] - np.ceil(waso[j]/time_interval)

    for j in range(len(real_sl_time)):
        real_wk_time[j]= 1 + 8 * (real_wk_time[j] - 1 )
        real_sl_time[j] = 1 + 8 * (real_sl_time[j] - 1 ) 
        real_wk_time_origin[j] = 1 + 8 * (real_wk_time_origin[j] - 1 )
        real_sl_time_origin[j] = 1 + 8 * (real_sl_time_origin[j] - 1 ) 

    tspan_temp = np.zeros(8*(len(tspan)-1))
    tspan_temp[0] = tspan[0]


    for j in range(len(tspan_temp)):
        tspan_temp[j] = tspan[0] + (j) * (time_interval/60)/8

    tspan = tspan_temp.copy()
    real_wk_time1 = real_wk_time.copy()
    real_wk_time1.insert(0, 1)
    st_fi = real_wk_time1.copy() 

    # % Between Initial and Day1
    ts = 24 * start_num

    it2 = np.arange(ts, 24*start_num+time_start, 
                    (24*start_num+time_start-ts)/(round(8*60/time_interval*(24*start_num+time_start-ts))+1))
    it3 = it2.copy()   

    for j in range(len(it2)):
        if it2[j] >= 24*(start_num):
            it3[j] = it3[j] - 24*start_num 

    i2 = np.zeros(len(it3))

    for j in range(len(it3)):
        if it3[j] >= 6:
            i2[j] = 250

    tspan2 = it2.copy()

    tspan_total = np.concatenate((tspan_first, tspan2[1:], tspan[1:]), axis=None)

    i_total = np.concatenate((i_first, i2[1:], np.array(light[1:])))                          
    it_total = np.concatenate((it_first, it2[1:], it[1:]))                           

    #% Simulation at the gap between the end time of intial 120 days and the
    #% starting time of actigraphy 
    it=it_total ; i = i_total 
    V_0 = y0[-1,:]

    t1_1 = tspan2.copy()
    y1_1 = odeint(func=PCR, y0=V_0, t=tspan2, args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)


    #% When users wear actiwatch very lately we assume no sleep after baseline
    #% sleep
    D_v = -10.2 - (3.37 * 0.5) * ( const + coef_y * y1_1[:,1] + coef_x * y1_1[:,0] ) + \
    v_vh * y1_1[:,5] 
    wakeup = np.where(D_v <= 2.46)[0]# 갯수 두개차이남
    if len(wakeup) != 0:
        if len(np.where(D_v[wakeup[0]:] > 2.46)[0]) != 0:
            sleep_re = np.where(D_v[wakeup[0]:] > 2.46)[0]
            V_0 =  [
                y1_1[wakeup[0]+sleep_re[0]-1,0], 
                y1_1[wakeup[0]+sleep_re[0]-1,1], 
                y1_1[wakeup[0]+sleep_re[0]-1,2], 
                wake_effort_Vv, 
                wake_effort_Vm, 
                y1_1[wakeup[0]+sleep_re[0]-1,5]
            ]
            it = it_total ; i = i_total 
            t1_1[wakeup[0]+sleep_re[0]-1:] = tspan2[wakeup[0]+sleep_re[0]-1:]
            y1_1[wakeup[0]+sleep_re[0]-1:,:] = odeint(func=PCR_shift, 
                                                      y0=V_0, t=t1_1[wakeup[0]+sleep_re[0]-1:], 
                                                      args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

    flag = 0                                                                
              
    y1_1 = pd.DataFrame(y1_1) 
    for day in range(len(st_fi)-1):
        # Simulation only with light data
        V_0 = y1_1.iloc[-1,:]  

        it=it_total ; i=i_total 

        t1_2 = tspan[int(st_fi[day])-1:int(st_fi[day+1])-1+1] 
        y1_2 = pd.DataFrame(
            odeint(func=PCR, y0=V_0, t=t1_2, 
                   args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)
        ) 
        
        a = 0
        while 1:
            a = a + 1
            matching_temp = Data_matching(day, tau_c, wake_effort_Vm, wake_effort_Vv, 
                                          sleep_effort_Vm, sleep_effort_Vv,  y1_2, st_fi, patt, 
                                          t1_2, tspan, real_sl_time, it_total,i_total)

            t1_2 = matching_temp.iloc[:,0] 
            y1_2 = matching_temp.iloc[:,1:] 
            y1_2.columns = [0,1,2,3,4,5]
            Qm = ( Q_max / (1+np.exp(-(y1_2.iloc[:,5-1]-theta)/sigma)) )

            # % Check the difference between data and simulation
            y_temp = np.repeat('NULL',int(1+(len(y1_2.iloc[:,6-1])-1)/8)).astype('U5')
            patt_simul = y_temp.copy()

            #len(patt_simul)

            for j in range(len(patt_simul)):
                if Qm[8*j] > 1:
                    patt_simul[j] = 'Wake'
                elif Qm[8*j] <=1:
                    patt_simul[j] = 'Sleep'


            diff_patt = np.where(patt[1+int((st_fi[day]+1-1)/8): 1+1+int((st_fi[day+1]-1)/8)]!= patt_simul)[0]            
            
            if len(diff_patt) == 0:
                break 

            if (diff_patt[0]+1) == len(patt_simul):
                break 

            if a == 10:
                print('Too much time, Error occurs')
                flag = 1
                break 

        if flag == 1:
            break 

        y1_1 = pd.concat([y1_1.iloc[:-1,:], y1_2], axis=0, ignore_index=True)
        t1_1 = pd.DataFrame(t1_1)
        t1_1 = pd.concat([t1_1.iloc[:-1,:], pd.DataFrame(t1_2)], axis=0,ignore_index=True)
    
    t_total = t1_1.iloc[len(tspan2)-1:,0]
    y_total = y1_1.iloc[len(tspan2)-1:,:]

    Q_m = ( Q_max / (1+np.exp(-(y_total.iloc[:,4]-theta)/sigma)) )

    sleep_model = np.where(Q_m <= 1)[0]
    sleep_change = np.where(np.diff(sleep_model)!=1)[0]

    sleep_start = np.append(0, sleep_change+1)
    sleep_end = np.append(sleep_change, len(sleep_model)-1)   

    H = y_total.iloc[:,5]
    C = (3.37 * 0.5) * ( const + coef_y * y_total.iloc[:,1] + coef_x * y_total.iloc[:,0] )

    D_v = -10.2 - (3.37 * 0.5) * ( const + coef_y * y_total.iloc[:,1] + coef_x * y_total.iloc[:,0] ) + v_vh * y_total.iloc[:,5]
    D_up = (2.46 + 10.2 + C)/v_vh

    my_xlim = [0,t_total.iloc[-1]-t_total.iloc[0]]
    my_ylim = [np.floor(min(H))-1, np.floor(max(H))+2]
    temp_tick = np.where(np.mod(t_total,6)==0)[0]

    if temp_tick[0] != 1:
        my_xtick = np.linspace(t_total.iloc[temp_tick[0]]-t_total.iloc[0],
                               t_total.iloc[temp_tick[-1]]-t_total.iloc[0],
                               len(temp_tick))
    else:
        my_xtick = np.linspace(t_total.iloc[temp_tick[1]]-t_total.iloc[temp_tick[0]],
                               t_total.iloc[temp_tick[-1]]-t_total.iloc[temp_tick[0]],
                               len(temp_tick)-1)

    my_ytick = np.linspace(np.floor(min(H)),np.floor(max(H))+1,3)[0]
    my_xlabel = np.zeros(len(my_xtick))

    for j in range(len(my_xlabel)): 
        my_xlabel[j] = np.mod(my_xtick[j]+t_total.iloc[0],24)


    # set figure
    #Fig = plt.figure()

    SMALL_SIZE = 8 
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 16 

    plt.rc('font', size=BIGGER_SIZE) # controls default text sizes 
    plt.rc('axes', titlesize=BIGGER_SIZE) # fontsize of the axes title 
    plt.rc('axes', labelsize=MEDIUM_SIZE) # fontsize of the x and y labels 
    plt.rc('xtick', labelsize=MEDIUM_SIZE) # fontsize of the tick labels 
    plt.rc('ytick', labelsize=MEDIUM_SIZE) # fontsize of the tick labels 
    plt.rc('legend', fontsize=BIGGER_SIZE) # legend fontsize 
    plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title
    plt.rcParams['axes.labelsize'] = MEDIUM_SIZE
    plt.rcParams['axes.labelweight'] = 'bold'

    lw = 5
    plt.figure(figsize=(30,10))
    plt.plot(t_total-t_total.iloc[0], D_up, 'y-',linewidth=lw)

    #hold on
    plt.plot(t_total-t_total.iloc[0], H, 'k-',linewidth=lw)
    plt.xlim(my_xlim)
    plt.ylim(my_ylim)

    for j in range(len(sleep_start)):
        plt.axvspan(t_total.iloc[sleep_model[sleep_start[j]]]-t_total.iloc[0] , 
                    t_total.iloc[sleep_model[sleep_end[j]]]-t_total.iloc[0],  
                    facecolor='skyblue', alpha=0.3)

    plt.savefig("graph.png", bbox_inches='tight', dpi=300)
    
    #% CSS
    #% Use first sleep time in 12:00-12:00 rather than 0:00-24:00
    sleep_time = tspan[real_sl_time[main_sleep[0][0]]] % 24
    
    if sleep_time < 12:
        sleep_time = sleep_time + 24
        
    sleep_time = sleep_time - 12
    
    #% [v] 에러 해결
    day_temp = np.floor((tspan[real_sl_time[main_sleep[0][-1]]] - tspan[real_sl_time[main_sleep[0][0]]]-(24.0-sleep_time))/24.0)    
    day_remain = (tspan[real_sl_time[main_sleep[0][-1]]] - tspan[real_sl_time[main_sleep[0][0]]] - (24-sleep_time)) % 24
    
    day_new = 1 + day_temp                                                 

    
    if day_remain > 0:
        day_new = day_new + 1
        
    base_of_sleep = np.floor((tspan[real_sl_time[main_sleep[0][0]]]-(24*start_num+12))/24)
         
    # We need to find the day which does not have main sleep 
    no_sleep = []
    if len(main_sleep[0]) < day_new:
        day_set = np.zeros(len(main_sleep[0]))
        for j in range(len(main_sleep[0])):
            day_set[j] = np.floor((tspan[real_sl_time[main_sleep[0][j]]]-
                                tspan[real_sl_time[main_sleep[0][0]]]-
                                (24-sleep_time))/24)+2
        for j in range(int(day_new)):
            trick_day = np.where(day_set==j+1)[0] 
            if len(trick_day) == 2:
                if (len(np.where(day_set==j)[0]) == 0 & len(np.where[day_set==j+1+1][0]) != 0):
                    day_set[trick_day[0]] = day_set[trick_day[0]] - 1
                elif (len(np.where(day_set==j)[0]) != 0 & len(np.where[day_set==j+1+1][0]) == 0):
                    day_set[trick_day[1]] = day_set[trick_day[1]] + 1
                else:
                    trick1 = (tspan[real_sl_time[main_sleep[0][trick_day[0]]]]-
                              tspan[real_sl_time[main_sleep[0][0]]]-(24-sleep_time)) % 24
                    trick2 = (tspan[real_sl_time[main_sleep[0][trick_day[1]]]]-
                              tspan[real_sl_time[main_sleep[0][0]]]-(24-sleep_time)) % 24
                    if trick1 <= 6:
                        day_set[trick_day[0]] = day_set[trick_day[0]] - 1
                    if trick2 >= 6+12:
                        day_set[trick_day[1]] = day_set[trick_day[1]] + 1
        for j in range(int(day_new)):
            if len(np.where(day_set==j+1)[0]) == 0:
                no_sleep.append(j)
    
    try:
        CSS, _, ness_sleep_amount = Calcul_CSS(len(no_sleep)+base_of_sleep, coef_x, coef_y, v_vh, tau_c, gate, main_sleep, y1_1, t1_1, tspan2, real_sl_time, real_wk_time, real_sl_time_origin, real_wk_time_origin)
        ness_sleep_amount = ness_sleep_amount[-1]
    except:
        CSS = -99
        ness_sleep_amount = -99

    return CSS, ness_sleep_amount