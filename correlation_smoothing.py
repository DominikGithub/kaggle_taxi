#!/usr/bin/env /home/dominik/anaconda2/bin/python

import pandas as pd
from utils_file import *
from utils_image import *

labels = ['weather', 'temp', 'pm25']

def poi_correlation():
    weekday = 0

    # pois = load(st.eval_dir + 'pois.bin')   #66x30
    # poi0 = np.repeat(pois[:,0], 144, axis=0).reshape((66, 144))
    # print poi0.shape

    demand = load(st.eval_dir + 'demand.bin')[:,:,:].reshape((66, 7*144))   #7x66x144
    print demand.shape

    supply = load(st.eval_dir + 'supply.bin')[:, :, :].reshape((66, 7*144))  # 7x66x144
    print supply.shape


    traffic_per_time = np.mean(load(st.eval_dir + 'traffic.bin'), axis=3)[:,:].reshape((66, 7*144))  # 7x66x144x4
    print traffic_per_time.shape

    pearR = np.corrcoef(demand, supply)
    # pearR = np.corrcoef(demand, poi0)

    # traf = load(st.eval_dir + 'traffic.bin')
    # print traf[0,:,:,0].shape
    # pearR = np.corrcoef(traf[0,:,:,0], traf[0,:,:,0])


    # pearR = np.corrcoef(traf[0,:,0,0], pois[:,0])
    print pearR
    print pearR.shape
    visualize_correlation(pearR, 'corr')

def weather_correlation(filename):
    weather = load(st.eval_dir + 'weather.bin')  # 21x144x3
    # weather = load(filename)
    print weather.shape

    pearR = np.corrcoef(weather[:,:,1], weather[:,:,2])
    print pearR
    print pearR.shape
    visualize_correlation(pearR, 'corr')

def smooth_weather():
    data = load(st.eval_dir+'weather.bin')
    n_days = 21
    name = 'weather'

    fig, axarr = plt.subplots(n_days, 3, figsize=(20, 30))
    smoothed = np.ndarray((n_days, 3, 144))
    for day in range(n_days):
        for objective in range(0, 3):
            S = pd.Series(data[day, :, objective])
            outlier_idx = S[S.pct_change() == np.inf].index.values-1
            outlier = S.copy()
            outlier[outlier_idx] = np.nan
            outlier = outlier.interpolate(method='pchip')

            df = pd.DataFrame(outlier)
            smoothed[day, objective, :] = df.rolling(window=4, center=True).median().fillna(0).values.flatten()

        axarr[day, 0].set_xticks(xrange(0, 144, 11))
        axarr[day, 0].plot(smoothed[day, 0], color='green')
        axarr[day, 1].set_xticks(xrange(0, 144, 11))
        axarr[day, 1].plot(smoothed[day, 1], color='red')
        axarr[day, 2].set_xticks(xrange(0, 144, 11))
        axarr[day, 2].plot(smoothed[day, 2], color='blue')

    save(st.eval_dir+name, smoothed)

    fig.suptitle('Weather, Temp, PM25', fontsize=20)
    plt.savefig(name+'.png')
    plt.close()

def smooth_weather_test():
    data = load(st.eval_dir_test+'weather.bin')
    n_days = 5
    name = 'weather_test'

    fig, axarr = plt.subplots(n_days, 3, figsize=(20, 10))
    smoothed = np.ndarray((n_days, 3, 144))
    for day in range(n_days):
        for objective in range(0, 3):
            smoothed[day, objective, :] = pd.rolling_max(data[day, :, objective], window=13, center=True)
            smoothed[day, objective, :] = pd.DataFrame(smoothed[day, objective, :]).fillna(0).values.flatten()
            # smoothed[day, objective, :] = df.rolling(window=4, center=True).median().fillna(0).values.flatten()     # clean nan's after median!!!

        axarr[day, 0].set_xticks(xrange(0, 144, 11))
        axarr[day, 0].plot(smoothed[day, 0], color='green')
        axarr[day, 1].set_xticks(xrange(0, 144, 11))
        axarr[day, 1].plot(smoothed[day, 1], color='red')
        axarr[day, 2].set_xticks(xrange(0, 144, 11))
        axarr[day, 2].plot(smoothed[day, 2], color='blue')

    save(st.eval_dir+name, smoothed)

    fig.suptitle('Weather, Temp, PM25', fontsize=20)
    plt.savefig(name+'.png')
    plt.close()

if __name__ == "__main__":
    # name = 'weather.bin'
    # weather_correlation(st.eval_dir+name)
    smooth_weather()
    # smooth_weather_test()

    # visualize_weather(data, 'Weather', '(Weather, Temp, PM25)')