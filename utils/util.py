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

def make_sleep_light(data): 

    time_interval = 2

    sleep_start_time = data.iloc[:,1] % 24
    sleep_end_time = data.iloc[:,3] % 24

    sleep_start_day = pd.to_datetime(data['sleep_start_day'])
    sleep_start_day  = (sleep_start_day-sleep_start_day[0])
    sleep_end_day = pd.to_datetime(data['sleep_end'])
    sleep_end_day = (sleep_end_day-sleep_end_day[0])

    if (sleep_end_time[0] < sleep_start_time[0]):
        sleep_end_day = sleep_end_day +  datetime.timedelta(days=1)

    time_start = np.floor(sleep_start_time[0]) - 1

    if time_start == 0:
        time_start = 24

    start_day = sleep_start_day[0]

    real_sl_time = sleep_start_day.apply(lambda x: x.total_seconds())/60/time_interval + round(sleep_start_time*60/time_interval)-time_start*60/time_interval+1
    real_wk_time = sleep_end_day.apply(lambda x: x.total_seconds())/60/time_interval + round(sleep_end_time*60/time_interval)-time_start*60/time_interval+1


    tspan = np.arange(start = time_start, 
                      stop = (time_start + (time_interval/60)* real_wk_time.tail(1)).item(), # 끝점 미포함
                      step = (time_interval/60), dtype=float)

    patt = np.repeat('Wake', len(tspan)).astype('U5')#, dtype='U25')

    tspan_sl_index = []
    for j in range(len(real_wk_time)): 
        tspan_sl_index.append(np.arange(real_sl_time[j],real_wk_time[j]))

    for idx in tspan_sl_index:
        patt[idx.astype(int)-1] = 'Sleep'

    patt_df = pd.DataFrame(patt, columns=['Sleep_pattern'])
    patt_df['light'] = np.where(patt_df['Sleep_pattern'] == 'Sleep', 0, 250)
    
    return patt_df


def PCR(y = None, t = None, it = None,i = None,tau_c = None,mu = None,
        v_vh = None,coef_x = None,coef_y = None,const = None,gate = None): 
    Q_max = 100
    
    theta = 10
    sigma = 3
    v_vm = 2.1
    v_mv = 1.8
    v_vc = 3.37
    A_m = 1.3
    A_v = - 10.2
    tau_m = 10 / (60 * 60)
    tau_v = 10 / (60 * 60)
    chi = 45
    Q_th = 1
    kappa = (12 / np.pi)
    i_0 = 9500
    p = 0.6
    gamma = 0.23
    
    k = 0.55
    beta = 0.013
    f = 0.99669
    alpha_0 = 0.16
    b = 0.4
    lambda_ = 60
    I = np.interp(t,it,i)
    dydt = np.zeros(6)
   
    dydt[0] = (1 / kappa) * ( gamma * ( y[0] - (4 * (y[0]**3)/3) ) - y[1] * ( ((24 / (f * tau_c))**2) + k * 19.9 * alpha_0 * (1-y[2]) * (1-b*y[0]) * (1-b*y[1]) * (
        (np.heaviside((Q_max / (1 + np.exp(-(y[4]-theta)/sigma)))-Q_th,1/2)+
         (np.heaviside((Q_max / (1 + np.exp(-(y[4]-theta)/sigma)))-Q_th,1/2)-1)*gate*(-0.03))**p*(I/i_0)**p) ) ) 

    dydt[1] = (1 / kappa) * \
    ( y[0] + 19.9 * alpha_0 * (1-y[2]) * (1-b*y[0]) * (1-b*y[1]) * \
     ((np.heaviside((Q_max / (1 + np.exp(-(y[4]-theta)/sigma)))-Q_th,1/2)+\
       (np.heaviside((Q_max / (1 + np.exp(-(y[4]-theta)/sigma)))-Q_th,1/2)-1)*gate*(-0.03))**p*(I/i_0)**p))
    
    dydt[2] = lambda_ * (alpha_0 * ((np.heaviside((Q_max / (1 + np.exp(-(y[4]-theta)/sigma)))-Q_th,1/2)+ \
                                    (np.heaviside((Q_max / (1 + np.exp(-(y[4]-theta)/sigma)))-Q_th,1/2)-1)* gate*(-0.03))**p*(I/i_0)**p) *\
                        (1-y[2]) - beta * y[2])
    dydt[3] = (1 / tau_v) * ( -y[3] - v_vm * ( Q_max / (1+np.exp(-(y[4]-theta)/sigma)) ) + A_v - (v_vc * (1/2) * (const + coef_y * y[1] + coef_x * y[0] ) ) + (v_vh * y[5]) )
    dydt[4] = (1 / tau_m) * ( -y[4] - v_mv * ( Q_max / (1+np.exp(-(y[3]-theta)/sigma)) ) + A_m)
    dydt[5] = (1 / chi) * (-y[5] + mu * ( Q_max / (1+np.exp(-(y[4]-theta)/sigma)) ) )
    
    return dydt

def PCR_shift(y = None, t = None, it = None,i = None,tau_c = None,mu = None,
        v_vh = None,coef_x = None,coef_y = None,const = None,gate = None): 
    
    Q_max = 100
    
    theta = 10
    sigma = 3
    v_vm = 2.1
    v_mv = 1.8
    v_vc = 3.37
    A_m = 1.3
    A_v = - 10.2
    tau_m = 10 / (60 * 60)
    tau_v = 10 / (60 * 60)
    chi = 45
    Q_th = 1
    
    kappa = (12 / np.pi)
    i_0 = 9500
    p = 0.6
    gamma = 0.23
    
    k = 0.55
    beta = 0.013
    f = 0.99669
    alpha_0 = 0.16
    b = 0.4
    lambda_ = 60
    I = np.interp(t,it,i)
    dydt = np.zeros(6)
   
    dydt[0] = (1 / kappa) * ( gamma * ( y[0] - (4 * (y[0]**3)/3) ) - y[1] * ( ((24 / (f * tau_c))**2) + k * 19.9 * alpha_0 * (1-y[2]) * (1-b*y[0]) * (1-b*y[1]) * (
        (np.heaviside((Q_max / (1 + np.exp(-(y[4]-theta)/sigma)))-Q_th,1/2)+
         (np.heaviside((Q_max / (1 + np.exp(-(y[4]-theta)/sigma)))-Q_th,1/2)-1)*gate*(-0.03))**p*(I/i_0)**p) ) ) 

    dydt[1] = (1 / kappa) * \
    ( y[0] + 19.9 * alpha_0 * (1-y[2]) * (1-b*y[0]) * (1-b*y[1]) * \
     ((np.heaviside((Q_max / (1 + np.exp(-(y[4]-theta)/sigma)))-Q_th,1/2)+\
       (np.heaviside((Q_max / (1 + np.exp(-(y[4]-theta)/sigma)))-Q_th,1/2)-1)*\
       gate*(-0.03))**p*(I/i_0)**p))
    
    dydt[2] = lambda_ * (alpha_0 * ((np.heaviside((Q_max / (1 + np.exp(-(y[4]-theta)/sigma)))-Q_th,1/2)+ \
                                    (np.heaviside((Q_max / (1 + np.exp(-(y[4]-theta)/sigma)))-Q_th,1/2)-1)* \
                                     gate*(-0.03))**p*(I/i_0)**p) *(1-y[2]) - beta * y[2])
    dydt[3] = 0
    dydt[4] = 0
    dydt[5] = (1 / chi) * (-y[5] + mu * ( Q_max / (1+np.exp(-(y[4]-theta)/sigma)) ) )
    
    return dydt

def Data_matching(day = None,tau_c = None,wake_effort_Vm = None,wake_effort_Vv = None,
                  sleep_effort_Vm = None,sleep_effort_Vv = None,y1_2 = None,
                  st_fi = None,patt = None,t1_2 = None,tspan = None,
                  real_sl_time = None,it_total = None,i_total = None): 
    Q_max = 100
    theta = 10
    sigma = 3
    mu = 4.2
    const = 1
    coef_x = - 0.16
    coef_y = 0.8
    v_vh = 1.01
    gate = 1

    it = it_total; i = i_total;
    # Calculate MA firing rate which decide whether the model state is wake or not

    t1_2 = pd.DataFrame(t1_2)
    y1_2 = pd.DataFrame(y1_2) 
    

    Qm = (Q_max / (1 + np.exp(- (y1_2.iloc[:,4] - theta) / sigma)))
    # Check the difference between data and simulation
    y_temp = np.zeros(int(1 + (len(y1_2.iloc[:,5])-1) / 8))
    patt_simul = y_temp.astype('str')
    for j in np.arange(len(patt_simul)):
        # Qm-firing rate of MA population-is closely related with arousal (Qm > 1 : awake)
        if Qm[ 8 * (j)] > 1:
            patt_simul[j] = 'Wake'
            # Qm <= 1 : asleep
        else:
            if Qm[8 * (j)] <= 1:
                patt_simul[j] = 'Sleep'
    
    # Compare Wake/Sleep state simulated by model with Wake/Sleep state from actigraphy
   
    
    patt_l = (patt[int((st_fi[day]-1)/ 8) : int((st_fi[day+1] -1) / 8) +1 ])
    patt_l = patt_l.reset_index().drop('index', axis=1)
    
    # [V] diff_patt 인덱스라서!  MATLAB에서의 값 -1
    diff_patt = np.where(patt_l[0] != pd.Series(patt_simul))[0]
    if len(diff_patt)==0: 
        # Last time point will be fixed at next wake-sleep block
        pass
    else:
        if diff_patt[0] != (len(patt_simul)-1):
            # When first dismatch occur, model is at the wake state
            if patt_simul[diff_patt[0]] == 'Wake':
                # If one-time point before dismatching has different sleep/wake 
                # state with that of dismatching find the exact time trasition of
                # wake state occur
                if patt_simul[diff_patt[0] -1] == 'Sleep':
                    Qm_temp = (Q_max /
                               (1 + 
                                np.exp(- 
                                       (y1_2.iloc[
                                           1 + (diff_patt[0] +1 -1) * 8 - 7-1:
                                           1 + (diff_patt[0] +1 -1) * 8 , 4] - 
                                        theta) / sigma)))
                    bifur = np.where(Qm_temp > 1)
                    diff_modi = 1 + 8 * (diff_patt[0] - 1) - 8 + bifur[0][0]
                else:
                    diff_modi = 1 + 8 * (diff_patt[0] )
                # Calculate Dv which decide whether model is in the bistable region or not
                Dv = - 10.2 - (3.37 * 0.5) * (const + coef_y * y1_2.iloc[diff_modi-1,2-1] + 
                                              coef_x * y1_2.iloc[diff_modi-1,1-1]) + v_vh * y1_2.iloc[diff_modi-1,6-1]
                # When model is in the bistable region
                if Dv >= 1.45:
                    # Change Vv,Vm as the value whenn the bifurcation occur
                    V_0 = [
                        y1_2.iloc[diff_modi,1-1],
                        y1_2.iloc[diff_modi,2-1],
                        y1_2.iloc[diff_modi,3-1],
                        0.0,
                        - 4.78,
                        y1_2.iloc[diff_modi,6-1]
                    ]
                    
                    it = it_total; i=i_total;
                    
                    
                    
                    t1_2.iloc[diff_modi-1:,1-1] = tspan[int(st_fi[day])+diff_modi-1-1:int(st_fi[day+1])]
                    y1_2.iloc[diff_modi-1:,:] = odeint(func=PCR, y0=V_0, t=t1_2.iloc[diff_modi-1:,1-1], 
                         args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15) 
                    
                    '''
                        [t1_2(diff_modi:end,1),y1_2(diff_modi:end,:)] = ode15s(@(t,V) ...
                        PCR(t,V,it_total,i_total,tau_c,mu, v_vh, ...
                            coef_x, coef_y, const, gate), ...
                                tspan(st_fi(day)+diff_modi-1:st_fi(day+1),1), V_0);
                    '''

                else: 
                    if patt_simul[diff_patt[0] - 1 -1] == 'Wake':
                        # Take the value of sleep effort as min of the Vm during
                        # 120th day of First 120 days-initialization
                        if t1_2.iloc[diff_modi,1-1] < t1_2.iloc[t1_2.shape[0] - 9-1,1-1]:
                            V_0 = [
                                    y1_2.iloc[diff_modi,1-1],
                                    y1_2.iloc[diff_modi,2-1],
                                    y1_2.iloc[diff_modi,3-1],
                                    sleep_effort_Vv,
                                    sleep_effort_Vm,
                                    y1_2.iloc[diff_modi,6-1]
                            ]

                            # Give sleep effort untill the time 'REST-S' of actigraphy is finished
                            t1_2.iloc[diff_modi:- 8+1,0] = tspan[int(st_fi[day] + diff_modi - 1):int(st_fi[day + 1] - 8+1)]
                            y1_2.iloc[diff_modi: - 8+1,:] =  odeint(func=PCR_shift, y0=V_0, t=t1_2.iloc[diff_modi:- 8+1,0], 
                             args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

                        else:
                            if t1_2.iloc[diff_modi,1-1] == t1_2.iloc[-9,0]:
                                V_0 = [
                                    y1_2.iloc[diff_modi,1-1],
                                    y1_2.iloc[diff_modi,2-1],
                                    y1_2.iloc[diff_modi,3-1],
                                    sleep_effort_Vv,
                                    sleep_effort_Vm,
                                    y1_2.iloc[diff_modi,6-1]
                            ]
                                
                                
                                # Give sleep effort untill the time 'REST-S' of actigraphy is finished
                                t_temp_error = np.arange(tspan[int(st_fi[day] + diff_modi - 1)],
                                                         tspan[int(st_fi[day + 1]) - 8], 
                                                         (tspan[int(st_fi[day + 1]) - 8]- 
                                                          tspan[int(st_fi[day] + diff_modi - 1)])/
                                                         (20-1))
                                
                               
                                
                                t_temp_err = t_temp_error.copy()
                                y_temp_err = odeint(func=PCR_shift, y0=V_0, t=pd.Series(t_temp_error),
                                 args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15) 

                                                            
                                t1_2.iloc[diff_modi,0] = t_temp_err[1-1]
                                t1_2.iloc[- 8,0] = t_temp_err[-1]
                                y1_2.iloc[diff_modi,:] = y_temp_err[0,:]
                                y1_2.iloc[- 8,:] = y_temp_err[-1,:]
                            else:
                                if t1_2.iloc[diff_modi,0] == t1_2.iloc[- 8,0]:
                                    y1_2.iloc[diff_modi,:] = [
                                                                y1_2.iloc[diff_modi,1-1],
                                                                y1_2.iloc[diff_modi,2-1],
                                                                y1_2.iloc[diff_modi,3-1],
                                                                sleep_effort_Vv,
                                                                sleep_effort_Vm,
                                                                y1_2.iloc[diff_modi,6-1]
                                                        ]
                                    
                        
                        # Stop using Sleep effort when the time 'REST-S' of actigraphy is finished
                        V_0 = [
                                y1_2.iloc[-8,1-1],
                                y1_2.iloc[-8,2-1],
                                y1_2.iloc[-8,3-1],
                                 y1_2.iloc[-8,4-1],
                                y1_2.iloc[-8,5-1],
                                y1_2.iloc[-8,6-1]
                        ]
                                    
                        t1_2.iloc[- 8: ,0] = \
                        tspan[int(st_fi[day + 1] - 8):int(st_fi[day + 1])]
                        
                        y1_2.iloc[- 8:,:] = \
                        odeint(func=PCR, y0=V_0, t=t1_2.iloc[- 8: ,0],
                                 args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

                    else:
                        if patt_simul[diff_patt[0] - 1] == 'Sleep':
                            # Take the value of sleep effort as min of the Vm during
                            # such interval
                            if t1_2.iloc[diff_modi,0] < t1_2.iloc[- 9,0]:
                                V_0 = [
                                        y1_2.iloc[diff_modi,1-1],
                                        y1_2.iloc[diff_modi,2-1],
                                        y1_2.iloc[diff_modi,3-1],
                                        sleep_effort_Vv,
                                        sleep_effort_Vm,
                                        y1_2.iloc[diff_modi,6-1]
                                ]
                                
                # Give sleep effort untill the time 'REST-S' of actigraphy is finished
                                t1_2.iloc[diff_modi: - 8,0] = tspan[int(st_fi[day]) + diff_modi - 1:
                                                                        int(st_fi[day + 1]) - 8]
                                y1_2.iloc[diff_modi:- 8+1,:] = \
                                odeint(func=PCR_shift, y0=V_0, t=t1_2.iloc[diff_modi-1: - 8,0],
                                 args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

                            else:
                                if (t1_2.iloc[diff_modi,0] == t1_2.iloc[- 9,0]):
                                    V_0 = [
                                        y1_2.iloc[diff_modi,1-1],
                                        y1_2.iloc[diff_modi,2-1],
                                        y1_2.iloc[diff_modi,3-1],
                                        sleep_effort_Vv,
                                        sleep_effort_Vm,
                                        y1_2.iloc[diff_modi,6-1]
                                ]

                                    
                                    # Give sleep effort untill the time 'REST-S' of actigraphy is finished
                                    t_temp_error = np.arange(tspan[int(st_fi[day] + diff_modi - 1)],
                                                         tspan[int(st_fi[day + 1]) - 8], 
                                                         (tspan[int(st_fi[day + 1]) - 8]- 
                                                          tspan[int(st_fi[day] + diff_modi - 1)])/
                                                         (20-1))
                                                
                                    t_temp_err = t_temp_error.copy()
                                    y_temp_err = odeint(func=PCR_shift, y0=V_0, t=pd.Series(t_temp_error),
                                     args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15) 
                    
                                    t1_2.iloc[-1,0] = t_temp_err[0]
                                    t1_2.iloc[- 8,0] = t_temp_err[-1]
                                    y1_2.iloc[diff_modi,:] = y_temp_err[0,:]
                                    y1_2.iloc[- 8,:] = y_temp_err[-1,:]
                                else:
                                    if t1_2.iloc[diff_modi,0] == t1_2.iloc[-8,0]:
                                        y1_2.iloc[diff_modi,:] = [
                                                y1_2.iloc[diff_modi,1-1],
                                                y1_2.iloc[diff_modi,2-1],
                                                y1_2.iloc[diff_modi,3-1],
                                                sleep_effort_Vv,
                                                sleep_effort_Vm,
                                                y1_2.iloc[diff_modi,6-1]
                                        ]
                            
                            # Stop using Sleep effort when the time 'REST-S' of actigraphy is finished
                            # When stop using sleep effort, change Vv, Vm as the value
                            # at the transition of sleep to wake occur
                            V_0 = [
                                y1_2.iloc[-8,1-1],
                                y1_2.iloc[-8,2-1],
                                y1_2.iloc[-8,3-1],
                                 y1_2.iloc[-8,4-1],
                                y1_2.iloc[-8,5-1],
                                y1_2.iloc[-8,6-1]
                        ]
                            t1_2.iloc[- 8:-1,0] = \
                            tspan[int(st_fi[day + 1] - 8):int(st_fi[day + 1])-1] 
    
                            y1_2.iloc[- 8:-1,:] = \
                            odeint(func=PCR, y0=V_0, t=t1_2.iloc[- 8:-1 ,0],
                                 args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)
               
        # When first dismatch occur, model is at the sleep state
            else:
                if patt_simul[diff_patt[0]] == 'Sleep':
                    # If one-time point before dismatching has different sleep/wake
                    # state with that of dismatching find the exact time trasition of
                    # sleep state occur
                    if diff_patt[0] != 0:
                        if patt_simul[diff_patt[0] - 1] == 'Wake':
                            Qm_temp = (Q_max / 
                                       (1 + 
                                        np.exp(- (
                                            y1_2.iloc[1 + (diff_patt[0] +1 - 1) * 8 - 7 -1: 
                                                      1 + (diff_patt[0] +1 - 1) * 8,5-1] - 
                                            theta) / sigma)))
                            bifur = np.where(Qm_temp <= 1)[0]
                            diff_modi = 1 + 8 * (diff_patt[0] +1 - 1) - 8 + bifur[0] 
                        else:
                            if patt_simul[diff_patt[0] - 1] == 'Sleep':
                                diff_modi = 1 + 8 * (diff_patt[0] - 1)
                    else:
                        diff_modi = 1 + 8 * (diff_patt[0] +1 - 1) -1
                    # Calculate Dv which decide whether model is in the bistable region
                    # or not
                    Dv = - 10.2 - (3.37 * 0.5) * (const + coef_y * y1_2.iloc[diff_modi,2-1] + \
                                                  coef_x * y1_2.iloc[diff_modi,0]) + \
    v_vh * y1_2.iloc[diff_modi,6-1]
                    # When model is in the bistable region
                    if Dv <= 2.46:
                        # Change Vv,Vm as the value whenn the bifurcation occur
                        V_0 = [
                                y1_2.iloc[diff_modi,1-1],
                                y1_2.iloc[diff_modi,2-1],
                                y1_2.iloc[diff_modi,3-1],
                                - 4.66,
                                - 0.12,
                                y1_2.iloc[diff_modi,6-1]
                                ]
                        t1_2.iloc[diff_modi:,0] = \
    tspan[int(st_fi[day] + diff_modi - 1):int(st_fi[day + 1])] 

                        y1_2.iloc[diff_modi:,:] =\
                        odeint(func=PCR, y0=V_0, t=t1_2.iloc[diff_modi:,0],
                             args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)                     

                    else:
                        if diff_patt[0] == 0:
                            # Take the value of wake effort as max (or min) of the Vm during
                            # 120th day of First 120 days-initialization
                            V_0 = [
                                y1_2.iloc[diff_modi,1-1],
                                y1_2.iloc[diff_modi,2-1],
                                y1_2.iloc[diff_modi,3-1],
                                wake_effort_Vv,
                                wake_effort_Vm,
                                y1_2.iloc[diff_modi,6-1]
                                ]
                            t1_2.iloc[diff_modi:,0] = tspan[int(st_fi[day] + diff_modi - 1):int(st_fi[day + 1])] 
        
                            y1_2.iloc[diff_modi:,:] = \
                            odeint(func=PCR_shift, y0=V_0, t=t1_2.iloc[diff_modi:,0] ,
                                   args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

                            # Give wake effort untill
                            # (a) When model enters the bistable region before 'Wake'
                            # state of actigraphy data finished
                            # (b) When 'Wake' time of actigraphy data finished
                            # Calculate Dv which decide whether model is in the bistable region
                            # or not
                            Dv = - 10.2 - (3.37 * 0.5) * \
    (const + coef_y * \
     y1_2.iloc[diff_modi : int(int(real_sl_time[day] - st_fi[day] + 1 - 8)+1),2-1] + \
     coef_x * y1_2.iloc[diff_modi:int(int(real_sl_time[day] - st_fi[day] + 1 - 8)+1),1-1]) + \
    v_vh * y1_2.iloc[diff_modi:int(int(real_sl_time[day] - st_fi[day] + 1 - 8)+1),6-1]
                            # Find interval when model enters the bistable region
                            # before 'Wake' state of actigraphy data finished
                            circa_align = np.where(Dv <= 2.46)[0]
                            # Case (a)
                            if len(circa_align) != 0:
                                circa_align_start = circa_align[0]
                                # When stop using wake effort, change Vv, Vm as the value
                                # at the transition of wake to sleep occur
                                V_0 =  [
                                    y1_2.iloc[diff_modi + circa_align_start - 1,1-1],
                                    y1_2.iloc[diff_modi + circa_align_start - 1,2-1],
                                    y1_2.iloc[diff_modi + circa_align_start - 1,3-1],
                                    -4.66,
                                    -0.12,
                                    y1_2.iloc[diff_modi + circa_align_start - 1,6-1]
                                ]

                                t1_2.iloc[diff_modi + circa_align_start :,0] =\
                                tspan[int(st_fi[day] + diff_modi + circa_align_start - 2):int(st_fi[day + 1])-1]
                                y1_2.iloc[diff_modi + circa_align_start :,:] = \
                                odeint(func=PCR, y0=V_0, t=t1_2.iloc[diff_modi + circa_align_start :,0],
                                   args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

                                # Case (b)
                            else:
                                V_0 =  [
                                        y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8),1-1],
                                        y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8),2-1],
                                        y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8),3-1],
                                        y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8),4-1],
                                        y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8),5-1],
                                        y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8),6-1]
                                        ]
                                t1_2.iloc[int(real_sl_time[day] - st_fi[day]+1 - 8)-1:,0] = \
                                tspan[int(real_sl_time[day]-8)-1:int(st_fi[day + 1])] 
                                y1_2.iloc[int(real_sl_time[day]-1-(st_fi[day])+1-8):,:] = \
                                odeint(func=PCR, y0=V_0, t=t1_2.iloc[int(real_sl_time[day] - st_fi[day]+1 - 8)-1:,0],
                                    args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

                        else:
                            if patt_simul[diff_patt[0] - 1] == 'Wake':
                                V_0 = [
                                    y1_2.iloc[diff_modi ,1-1],
                                    y1_2.iloc[diff_modi ,2-1],
                                    y1_2.iloc[diff_modi ,3-1],
                                    wake_effort_Vv,
                                    wake_effort_Vm,
                                    y1_2.iloc[diff_modi ,6-1]
                                ]

                                t1_2.iloc[diff_modi:,0] = \
                                tspan[int(st_fi[day] + diff_modi - 1):int(st_fi[day + 1])]
                                
                                y1_2.iloc[diff_modi:,:] = \
                                odeint(func=PCR_shift, y0=V_0, t=t1_2.iloc[diff_modi:,0],
                                   args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

                                
                                
                                # Give wake effort untill
                                # (a) When model enters the bistable region before 'Wake'
                                # state of actigraphy data finished
                                # (b) When 'Wake' time of actigraphy data finished
                                # Calculate Dv which decide whether model is in the bistable region
                                # or not
                                Dv = - 10.2 - (3.37 * 0.5) * \
    (const + coef_y * 
     y1_2.iloc[diff_modi : int(int(real_sl_time[day] - st_fi[day] + 1 - 8)),2-1] + \
     coef_x * y1_2.iloc[diff_modi:int(int(real_sl_time[day] - st_fi[day] + 1 - 8)),1-1]) + \
    v_vh * y1_2.iloc[diff_modi:int(int(real_sl_time[day] - st_fi[day] + 1 - 8)),6-1]
                                # Find interval when model enters the bistable region
                                # before 'Wake' state of actigraphy data finished
                                circa_align = np.where(Dv <= 2.46)[0]
                                # Case (a)
                                if len(circa_align)!=0:
                                    circa_align_start = circa_align[0]
                                    # When stop using wake effort, change Vv, Vm as the value
                                    # at the transition of wake to sleep occur
                                    V_0 =  [
                                        y1_2.iloc[diff_modi + circa_align_start - 1,1-1],
                                        y1_2.iloc[diff_modi + circa_align_start - 1,2-1],
                                        y1_2.iloc[diff_modi + circa_align_start - 1,3-1],
                                        y1_2.iloc[diff_modi + circa_align_start - 1,4-1],
                                        y1_2.iloc[diff_modi + circa_align_start - 1,5-1],
                                        y1_2.iloc[diff_modi + circa_align_start - 1,6-1]
                                    ]

                                    t1_2.iloc[diff_modi + circa_align_start :,0] =\
                                    tspan[int(st_fi[day] + diff_modi + circa_align_start - 2):int(st_fi[day + 1])-1]
                                    y1_2.iloc[diff_modi + circa_align_start :,:] = \
                                    odeint(func=PCR, y0=V_0, t=t1_2.iloc[diff_modi + circa_align_start :,0],
                                       args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

                               # Case (b)
                                else:
                                    V_0 =  [
                                            y1_2.iloc[int(int(real_sl_time[day] - st_fi[day] + 1 - 8)),1-1],
                                            y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8),2-1],
                                            y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8),3-1],
                                            y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8),4-1],
                                            y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8),5-1],
                                            y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8),6-1]
                                        ]
                                    t1_2.iloc[int(real_sl_time[day]- (st_fi[day]+1) + 1 - 8):,0] = \
                                    tspan[int(real_sl_time[day]-8)-1:int(st_fi[day + 1])] 
                                    y1_2.iloc[int(real_sl_time[day]-1-(st_fi[day])+1-8):,:] = \
                                    odeint(func=PCR, y0=V_0, t=t1_2.iloc[int(real_sl_time[day] - (st_fi[day]+1) + 1 - 8):,0],
                                       args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

    
    matching_data = pd.concat([t1_2,y1_2], axis=1)
    return matching_data
    
def Calcul_CSS(no_sleep = None,coef_x = None,coef_y = None,v_vh = None,tau_c = None,gate = None,main_sleep = None,y1_1 = None,t1_1 = None,tspan2 = None,real_sl_time = None,real_wk_time = None,real_sl_time_origin = None,real_wk_time_origin = None): 
    
    Q_max = 100
    theta = 10
    sigma = 3
    mu = 4.2
    const = 1
    
    y1_1 = pd.DataFrame(y1_1)
    # Calculate Dv to determine sufficient or not
    D_v = - 10.2 - (3.37 * 0.5) * \
    (const + coef_y * y1_1.iloc[len(tspan2)-1:,1] + \
     coef_x * y1_1.iloc[len(tspan2)-1:,0]) + v_vh * y1_1.iloc[len(tspan2)-1:,5]
    CSS_Dv = np.zeros((len(real_wk_time)))
    
    CSS_temp = np.zeros(len(main_sleep[0]))
    
    CSS_temp2 = np.zeros(len(main_sleep[0]))
    
    for j in range(len(real_wk_time)):
        CSS_Dv[j] = D_v.iloc[int(real_wk_time[j])-1]
    
    for j in range(len(main_sleep[0])):
        CSS_temp2[j] = CSS_Dv[main_sleep[0][j]]
    
    early_confirm = np.zeros(len(main_sleep[0]))
    
    for j in range(len(main_sleep[0])):
        confirm_early = np.where(D_v.iloc[int(real_sl_time[main_sleep[0][j]]):int(real_wk_time[main_sleep[0][j]]+1)] > 2.46)[0]
        if (len(confirm_early) == 0):
            early_confirm[j] = 1
    
    y_temp = y1_1.iloc[len(tspan2)-1:,:]
    time_temp = t1_1.iloc[len(tspan2)-1:,:]
    necc_sleep_amount = np.zeros(len(main_sleep[0]))
    
    for j in range(len(main_sleep[0])):
        if early_confirm[j] == 1:
            V_0 = y_temp.iloc[real_sl_time[main_sleep[0][j]] - 8,:]
            time_early = np.zeros(8 * 30 * 24)
            for t_index in range(len(time_early)):
                time_early[t_index] = time_temp.iloc[real_sl_time[main_sleep[0][j]] - 8,:] + 2 / (60 * 8) * (t_index - 1)
            i_early = 250 * np.ones(8 * 30 * 24)
            t_early = time_early.copy() 
            it = time_early; i=i_early;
            y_early = odeint(func=PCR, y0=V_0, t=time_early, 
               args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)
            Q_temp = (Q_max / (1 + np.exp(- (y_early[:,4] - theta) / sigma)))
            sleep_start = np.where(Q_temp <= 1)[0]
            V_0 = y_early[sleep_start[0],:]
            time_temp2 = t_early[sleep_start[0]]
            time_early2 = np.zeros(8 * 30 * 24)
            for t_index in range(len(time_early2)):
                time_early2[t_index] = time_temp2 + 2 / (60 * 8) * (t_index - 1)
            i_early2 = np.zeros(8 * 30 * 24)
            
            it=time_early2; i = i_early2;
            
            t_early2 = time_early2.copy()
            y_early2 = odeint(func=PCR, y0=V_0, t=time_early2, 
               args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)
            
            Dv_temp = - 10.2 - (3.37 * 0.5) * (const + coef_y * y_early2[:,1] + coef_x * y_early2[:,0]) + v_vh * y_early2[:,5]
            wake_start = np.where(Dv_temp <= 2.46)[0]
            necc_sleep_amount[j] = (wake_start[0] - 1) * 2 / (60 * 8)
        else:
            if CSS_temp2[j] <= 2.46:
                Dv_temp = D_v[real_sl_time[main_sleep[0][j]]:real_wk_time[ain_sleep[j]]+1]
                over_sleep_thres = np.where(Dv_temp <= 2.46)[0]
                early_normal = np.where(np.diff(over_sleep_thres) != 1)[0]
                #This kind of difference can be happened
                if len(early_normal)==0:
                    necc_sleep_amount[j] = (over_sleep_thres[0] - 1) * 2 / (60 * 8)
                else:
                    necc_sleep_amount[j] = (over_sleep_thres(early_normal[0] + 1) - 1) * 2 / (60 * 8)
            else:
                V_0 = y_temp[real_wk_time[main_sleep[0][j]] - 8,:]
                time_more = np.zeros(8 * 30 * 24)
                for t_index in range(len(time_more)):
                    time_more[t_index] = time_temp[real_wk_time[main_sleep[0][j]] - 8,:] + 2 / (60 * 8) * (t_index - 1)
                i_more = np.zeros(8 * 30 * 24)
                
                i=i_more; it=time_more;
                t_more = time_more.copy()
                y_more =  odeint(func=PCR, y0=V_0, t=time_more, 
                                 args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)
                #ode15s(lambda t = None,V = None: 
                #PCR(t,V,time_more,i_more,tau_c,mu,v_vh,coef_x,coef_y,const,gate),
                #time_more,V_0)
                Dv_temp = - 10.2 - (3.37 * 0.5) * (const + coef_y * y_more[:,1] + coef_x * y_more[:,0]) + v_vh * y_more[:,5]
                wake_start = np.where(Dv_temp <= 2.46)[0]
                necc_sleep_amount[j] = (wake_start[0] - 9) * 2 / (60 * 8) + (real_wk_time[main_sleep[0][j]] - real_sl_time[main_sleep[0][j]]) * 2 / (60 * 8)
                #Calculate necessary sleep = Actual sleep duration + more needed time
    
                # For early sleep, sleep duration must be larger than normal sleep amount
                # For the others, Dv<=2.46 is enough for being sufficient sleep
    for j in range(len(main_sleep[0])):
        if early_confirm[j] == 1:
            if (real_wk_time[main_sleep[0][j]] - real_sl_time[main_sleep[0][j]]) * 2 / (60 * 8) > necc_sleep_amount[j]:
                CSS_temp[j] = 1
        else:
            if CSS_Dv[main_sleep[0][j]] <= 2.46:
                CSS_temp[j] = 1
    
    for j in range(len(main_sleep[0])):
        if j != (len(main_sleep[0])-1):
            if (CSS_temp[j] == 0) & (main_sleep[0][j] + 1 < main_sleep[0][j + 1]):
                near_range = np.array(real_sl_time_origin[main_sleep[0][j]:main_sleep[0][j + 1] - 1+1]) - real_wk_time_origin[main_sleep[0][j]]
                near_main = np.where(near_range * 2 / (60 * 8) <= 3)[0]
                sl_du = (np.array(real_wk_time[main_sleep[0][j] + near_main[0] - 1:
                                      main_sleep[0][j] + near_main[-1] - 1+1]) - \
                         np.array(real_sl_time[main_sleep[0][j] + near_main[0] - 1 : 
                                      main_sleep[0][j] + near_main[-1] - 1+1])) * 2 / (60 * 8)
                if necc_sleep_amount[j] <= sum(sl_du):#sum(sum(sl_du)):
                    CSS_temp[j] = 1
        else:
            if (CSS_temp[j] == 0) & (main_sleep[0][j] < len(real_sl_time)):
                near_range = real_sl_time_origin[main_sleep[0][j]:len(real_sl_time)+1] - \
                real_wk_time_origin[main_sleep[0][j]]
                near_main = np.where(near_range * 2 / (60 * 8) <= 3)[0]
                sl_du = (np.array(real_wk_time[main_sleep[0][j] + near_main[0] - 1 : main_sleep[0][j] + \
                                      near_main[-1] - 1+1]) - \
                         np.array(real_sl_time[main_sleep[0][j] + near_main[0] - 1: 
                                      main_sleep[0][j] + near_main[-1] - 1+1])) * 2 / (60 * 8)
                if necc_sleep_amount[j] <= sum(sl_du):#sum(sum(sl_du)):
                    CSS_temp[j] = 1
    
    CSS = 100 * (sum(CSS_temp)) / (len(main_sleep[0]) + no_sleep)

    return CSS, CSS_temp, necc_sleep_amount


def main(sleep_df, waso_df):

    time_interval = 2; 
    time_start = 24;   
    tau_c = 24.09;  

    patt = sleep_df.iloc[:,0]
    light = sleep_df.iloc[:,1]

    if patt.tail(1).item()=='Sleep':
        patt = patt.append(pd.Series('Wake'),ignore_index=True)
        light = light.append(pd.Series(250), ignore_index=True)

    waso = waso_df.iloc[:,0]
    main_sleep_temp = waso_df.iloc[:,1]                       
    main_sleep = np.where(main_sleep_temp=='M')


    Q_max = 100; theta = 10; sigma = 3; 
    mu = 4.2; coef_y = 0.8; const = 1; v_vh = 1.01; coef_x = -0.16;                
    gate = 1;      

    start_num = 120;

    if time_start < 12:
        time_start = time_start + 24


    it_first= np.linspace(0, 24*start_num, (24*start_num)*30+1);       
    it2 = it_first.copy()

    for k in range(len(it2)):
        it2[k] =  it2[k] - 24 * (np.floor(it2[k]/24));
    # Light on from 6 a.m. to 21:42 a.m. as 250 lux [*]:= i_first                                                                      
    i_first = np.zeros(len(it_first))   
    # Reasonable light on timing according to
    # baseline sleep-wake schedule (sleep occurs between 22:00-6:00) 
    for j in range(len(i_first)):                                                 
        if (it2[j] >= 6) & (it2[j] < 22-0.3):                                       
            i_first[j] = 250;


    tspan_first = it_first.copy()                                                  

    V_00 =  [-0.5, -0.25, 5, 0, -11, 14.25]


    it = it_first; i=i_first; 
    sol = odeint(func=PCR, y0=V_00, t=tspan_first, 
                 args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

    y0 = sol; t0 = tspan_first

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
            patt.loc[int(real_wk_time[j] - np.ceil(waso[j]/time_interval)) : int(real_wk_time[j])] = 'Wake'
            real_wk_time[j] = real_wk_time[j] - np.ceil(waso[j]/time_interval)

    for j in range(len(real_sl_time)):
        real_wk_time[j]= 1 + 8 * (real_wk_time[j] - 1 + 1)
        real_sl_time[j] = 1 + 8 * (real_sl_time[j] - 1 + 1) 
        real_wk_time_origin[j] = 1 + 8 * (real_wk_time_origin[j] - 1 +1 )
        real_sl_time_origin[j] = 1 + 8 * (real_sl_time_origin[j] - 1 +1) 

    tspan_temp = np.zeros(8*(len(tspan)-1) + 1)
    tspan_temp[0] = tspan[0]


    for j in range(len(tspan_temp)):
        tspan_temp[j] = tspan[0] + (j) * (time_interval/60)/8

    tspan = tspan_temp.copy()
    real_wk_time.insert(0, 1)
    st_fi = real_wk_time.copy() 

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
    it=it_total; i = i_total;
    V_0 = y0[-1,:]


    t1_1 = tspan2.copy()
    y1_1 = odeint(func=PCR, y0=V_0, t=tspan2, 
                  args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)


    #% When users wear actiwatch very lately we assume no sleep after baseline
    #% sleep
    D_v = -10.2 - (3.37 * 0.5) * ( const + coef_y * y1_1[:,1] + coef_x * y1_1[:,0] ) + \
    v_vh * y1_1[:,5];
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
            it = it_total; i = i_total;
            t1_1[wakeup[0]+sleep_re[0]-1:] = tspan2[wakeup[0]+sleep_re[0]-1:]
            y1_1[wakeup[0]+sleep_re[0]-1:,:] = odeint(func=PCR_shift, 
                                                      y0=V_0, t=t1_1[wakeup[0]+sleep_re[0]-1:], 
                                                      args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)

            #tspan2(wakeup[0]+sleep_re[0]-1:end), V_0)      

    flag = 0                                                                

    # %Dummy index for checking match between real and predicted sleep-                       
    y1_1 = pd.DataFrame(y1_1) 

    for day in range(len(st_fi)-1):
        # Simulation only with light data
        V_0 = y1_1.iloc[-1,:]  


        it=it_total; i=i_total;

        t1_2 = tspan[int(st_fi[day])-1:int(st_fi[day+1])-1+1] 
        y1_2 = pd.DataFrame(
            odeint(func=PCR, y0=V_0, t=t1_2, 
                   args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)
        ) 


        a= 0

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


            diff_patt = np.where(patt[int((st_fi[day]+1-1)/8): int((st_fi[day]+1-1)/8)+len(patt_simul)]!= patt_simul)[0] 

            if len(diff_patt) == 0:
                break

            if (diff_patt[0]+1) == len(patt_simul):
                break

            if a == 10:
                print('Too much time, Error occurs')
                flag = 1
                break
                #return -100, -100

        if flag == 1:
            break;
        y1_1 = pd.concat([y1_1.iloc[:-1,:], y1_2], axis=0,ignore_index=True)
        t1_1 = pd.DataFrame(t1_1)
        t1_1 = pd.concat([t1_1.iloc[:-1,:], pd.DataFrame(t1_2)], axis=0,ignore_index=True)

    #%% H_C.png  

    Font_size = 15
    Font_weight = 'bold'
    axis_thick = 1.5



    t_total = t1_1.iloc[len(tspan2)-1:,0]
    y_total = y1_1.iloc[len(tspan2)-1:,:]

    Q_m = ( Q_max / (1+np.exp(-(y_total.iloc[:,4]-theta)/sigma)) )

    sleep_model = np.where(Q_m <= 1)[0]
    sleep_change = np.where(np.diff(sleep_model)!=1)[0]


    sleep_start = np.append(0, sleep_change+1)
    sleep_end = np.append(sleep_change, len(sleep_model)-1)   

    H = y_total.iloc[:,5];
    C = (3.37 * 0.5) * ( const + coef_y * y_total.iloc[:,1] + coef_x * y_total.iloc[:,0] );

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
                               len[temp_tick]-1)

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
    plt.plot(t_total-t_total.iloc[0], D_up, 'y-',linewidth=lw)#, label='theta(t)') 
    #h1 = plot(t_total-t_total(1),D_up,'Color',[255,194,71]/255, 'LineWidth',lw);
    #hold on
    plt.plot(t_total-t_total.iloc[0],H, 'k-',linewidth=lw)
    #h2 = plot(t_total-t_total(1),H,'Color','k', 'LineWidth',lw);

    for j in range(len(sleep_start)):
        plt.axvspan(t_total.iloc[sleep_model[sleep_start[j]]]-t_total.iloc[0] , 
                    t_total.iloc[sleep_model[sleep_end[j]]]-t_total.iloc[0],  
                    facecolor='skyblue', alpha=0.3)

    plt.savefig("./temp/graph.png", bbox_inches='tight', dpi=300)
    try:
        CSS, _, ness_sleep_amount = Calcul_CSS(0, coef_x, coef_y, v_vh, tau_c, gate, main_sleep, y1_1, t1_1, tspan2, real_sl_time, real_wk_time, real_sl_time_origin, real_wk_time_origin)
        ness_sleep_amount = ness_sleep_amount[-1]
    except:
        CSS = -99
        ness_sleep_amount = -99

    return CSS, ness_sleep_amount