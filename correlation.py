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
    visualize_poi_correlation(pearR, 'corr')

def weather_correlation():
    weather = load(st.eval_dir + 'weather.bin')  # 21x144x3
    print weather.shape

    pearR = np.corrcoef(weather[:,:,0], weather[:,:,1])
    print pearR
    print pearR.shape
    visualize_correlation(pearR, 'corr')

def smooth_weather(data):
    fig, axarr = plt.subplots(21, 3, figsize=(20, 30))


    for day in range(21):
        smoothed = np.ndarray((3, 144))
        for objective in range(0, 3):
            S = pd.Series(data[day,:,objective])
            outlier_idx = S[S.pct_change() == np.inf].index.values-1
            outlier = S.copy()
            outlier[outlier_idx] = np.nan
            outlier = outlier.interpolate(method='pchip')

            df = pd.DataFrame(outlier)
            smoothed[objective,:] = df.rolling(window=4, center=True).median().values.flatten()

        axarr[day, 0].set_xticks(xrange(0, 144, 11))
        axarr[day, 0].plot(smoothed[0], color='green')
        axarr[day, 1].set_xticks(xrange(0, 144, 11))
        axarr[day, 1].plot(smoothed[1], color='red')
        axarr[day, 2].set_xticks(xrange(0, 144, 11))
        axarr[day, 2].plot(smoothed[2])


    fig.suptitle(labels[objective], fontsize=20)
    plt.savefig('smoothed.png')
    plt.close()


if __name__ == "__main__":
    # weather_correlation()

    data = load(st.eval_dir + 'weather.bin')
    smooth_weather(data)

    # visualize_weather(data, 'Weather', '(Weather, Temp, PM25)')