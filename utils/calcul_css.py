import pandas as pd
import numpy as np
from scipy.integrate import odeint
from utils.PCR import PCR
from utils.PCR_shift import PCR_shift

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
            it = time_early ; i=i_early 
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
            
            it=time_early2 ; i = i_early2 
            
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
                
                i=i_more ; it=time_more 
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
