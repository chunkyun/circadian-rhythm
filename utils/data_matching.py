import pandas as pd
from pandas import Series, DataFrame
import math
import numpy as np
from scipy import stats
from scipy.integrate import odeint

from utils.PCR_shift import PCR_shift
from utils.PCR import PCR


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

    it = it_total ; i = i_total 
    # Calculate MA firing rate which decide whether the model state is wake or not

    t1_2 = pd.DataFrame(t1_2)
    y1_2 = pd.DataFrame(y1_2) 
    
    Qm = (Q_max / (1 + np.exp(- (y1_2.iloc[:,4] - theta) / sigma)))
    # Check the difference between data and simulation
    y_temp = np.zeros(int(1 + (len(y1_2.iloc[:,5])-1) / 8))
    patt_simul = y_temp.astype('str')
    for j in range(len(patt_simul)):
        # Qm-firing rate of MA population-is closely related with arousal (Qm > 1 : awake)
        if Qm[ 8 * (j)] > 1:
            patt_simul[j] = 'Wake'
            # Qm <= 1 : asleep
        else:
            if Qm[8 * (j)] <= 1:
                patt_simul[j] = 'Sleep'

    # Compare Wake/Slep state simulated by model with Wake/Sleep state from actigraphy
   
    
    patt_l = (patt[int((st_fi[day]-1)/ 8)+1: int((st_fi[day+1] -1) / 8) +1 +1])
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
                    Qm_temp = (Q_max / (1 + np.exp(- (y1_2.iloc[1 + (diff_patt[0] +1 -1) * 8 - 7-1:
                                        1 + (diff_patt[0] +1 -1) * 8 , 4] - theta) / sigma)))
                    bifur = np.where(Qm_temp > 1)
                    diff_modi = 1 + 8 * (diff_patt[0] +1 -1) - 8 + bifur[0][0]
                else:
                    diff_modi = 1 + 8 * (diff_patt[0] + 1 -1) -1
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
                    
                    it = it_total ; i=i_total 
                    
                    t1_2.iloc[diff_modi-1:,1-1] = tspan[int(st_fi[day])+diff_modi-1-1:int(st_fi[day+1])]
                    y1_2.iloc[diff_modi-1:,:] = odeint(func=PCR, y0=V_0, t=t1_2.iloc[diff_modi-1:,1-1], 
                         args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15) 
                    

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
                                t1_2.iloc[-8,0] = t_temp_err[-1]
                                y1_2.iloc[diff_modi,:] = y_temp_err[0,:]
                                y1_2.iloc[-8,:] = y_temp_err[-1,:]
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
                        V_0 = [
                                y1_2.iloc[-8,1-1],
                                y1_2.iloc[-8,2-1],
                                y1_2.iloc[-8,3-1],
                                 y1_2.iloc[-8,4-1],
                                y1_2.iloc[-8,5-1],
                                y1_2.iloc[-8,6-1]
                        ]
                                    
                        t1_2.iloc[-8: ,0] = \
                        tspan[int(st_fi[day + 1] - 8):int(st_fi[day + 1])]
                        
                        y1_2.iloc[-8:,:] = \
                        odeint(func=PCR, y0=V_0, t=t1_2.iloc[-8: ,0],
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
                                t1_2.iloc[diff_modi:-8,0] = tspan[int(st_fi[day]) + diff_modi - 1:
                                                                        int(st_fi[day + 1]) - 8]
                                y1_2.iloc[diff_modi:-8+1,:] = \
                                odeint(func=PCR_shift, y0=V_0, t=t1_2.iloc[diff_modi-1: - 8,0],
                                 args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)
                                print("y1_2")
                                print(y1_2)
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
                                y1_2.iloc[-8,6-1] ]
                            t1_2.iloc[-8:,0] = tspan[int(st_fi[day + 1] - 8):int(st_fi[day + 1])] 
                            y1_2.iloc[- 8:,:] = odeint(func=PCR, y0=V_0, t=t1_2.iloc[-8: ,0],
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
                                diff_modi = 1 + 8 * (diff_patt[0] +1 - 1) -1
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
                        t1_2.iloc[diff_modi:,0] = tspan[int(st_fi[day] + diff_modi - 1):int(st_fi[day + 1])] 

                        y1_2.iloc[diff_modi:,:] = odeint(func=PCR, y0=V_0, t=t1_2.iloc[diff_modi:,0],
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
                                            y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8 -1),1-1],
                                            y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8 -1),2-1],
                                            y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8 -1),3-1],
                                            y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8 -1),4-1],
                                            y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8 -1),5-1],
                                            y1_2.iloc[int(real_sl_time[day] - st_fi[day] + 1 - 8 -1),6-1]
                                        ]
                                    t1_2.iloc[int(real_sl_time[day]- (st_fi[day]+1) + 1 - 8):,0] = \
                                    tspan[int(real_sl_time[day]-8)-1:int(st_fi[day + 1])] 
                                    y1_2.iloc[int(real_sl_time[day]-1-(st_fi[day])+1-8):,:] = \
                                    odeint(func=PCR, y0=V_0, t=t1_2.iloc[int(real_sl_time[day] - (st_fi[day]+1) + 1 - 8):,0],
                                       args=(it,i,tau_c,mu, v_vh, coef_x, coef_y, const, gate), mxords=15)
    matching_data = pd.concat([t1_2, y1_2], axis=1)
    return matching_data