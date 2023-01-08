import pandas as pd
import numpy as np
import datetime

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
                      stop = (time_start + (time_interval/60)* real_wk_time.tail(1)).item() + 0.0001, # 끝점 미포함
                      step = (time_interval/60), dtype=float)

    patt = np.repeat('Wake', len(tspan)).astype('U5')

    tspan_sl_index = []
    for j in range(len(real_wk_time)): 
        tspan_sl_index.append(np.arange(real_sl_time[j],real_wk_time[j]))

    for idx in tspan_sl_index:
        patt[idx.astype(int)-1] = 'Sleep'

    patt_df = pd.DataFrame(patt, columns=['Sleep_pattern'])
    patt_df['light'] = np.where(patt_df['Sleep_pattern'] == 'Sleep', 0, 250)    
    
    return patt_df