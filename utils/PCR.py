import numpy as np

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

