#!/usr/bin/env /home/dominik/anaconda2/bin/python

import pandas as pd
from utils_file import *
from utils_image import *

labels = ['weather', 'temp', 'pm25']

def poi_correlation():
    # pois = load(st.eval_dir + 'pois.bin')   #66x30
    # poi0 = np.repeat(pois[:,0], 144, axis=0).reshape((66, 144))
    # print poi0.shape

    gap_daywise = load(st.eval_dir + 'gap_daywise.bin')[:,:,:].reshape((66, 22*144))   #22x66x144
    print gap_daywise.shape

    supply_daywise = load(st.eval_dir + 'supply_daywise.bin')[:, :, :].reshape((66, 22*144))  # 22x66x144
    print supply_daywise.shape

    traffic_per_time = np.mean(load(st.eval_dir + 'traffic_daywise.bin'), axis=3)[:,:].reshape((66, 22*144))  #22x66x144x4
    print traffic_per_time.shape

    pearR = np.corrcoef(gap_daywise, traffic_per_time)
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

def interpolate_traffic(district_idx):
    week = load(st.eval_dir + 'traffic.bin')
    all_days = np.ndarray((7, st.n_districts, 144, 4))
    idx = 0
    for day in week:
        clean_data = np.delete(day, district_idx, axis=0)
        mean_traffic = np.sum(clean_data, axis=0, keepdims=True) / len(clean_data)
        first = day[:district_idx,:,:]
        second = day[district_idx+1:,:,:]
        all_days[idx] = np.concatenate((first, mean_traffic, second), axis=0)
        idx += 1
    return all_days

def smooth_visualize_weather_train():
    print 'smoothing weather training data ...'
    name = 'weather_daywise'
    data = load(st.eval_dir+'weather.bin')
    n_days = 31

    fig, axarr = plt.subplots(n_days, 3, figsize=(20, 30))
    plt.ioff()
    smoothed = np.ndarray((n_days, 3, 144))
    for day in range(n_days):
        for objective in range(0, 3):
            S = pd.Series(data[day, :, objective])
            outlier_idx = S[S.pct_change() == np.inf].index.values-1
            outlier = S.copy()
            outlier[outlier_idx] = np.nan
            outlier = outlier.interpolate(method='pchip')

            df = pd.DataFrame(outlier)
            temp = df.rolling(window=4, center=True).median().fillna(0).values.flatten()
            smoothed[day, objective, :] = temp

        axarr[day, 0].set_xticks(xrange(0, 144, 11))
        axarr[day, 0].plot(smoothed[day, 0], color='green')
        axarr[day, 1].set_xticks(xrange(0, 144, 11))
        axarr[day, 1].plot(smoothed[day, 1], color='red')
        axarr[day, 2].set_xticks(xrange(0, 144, 11))
        axarr[day, 2].plot(smoothed[day, 2], color='blue')

    save(st.eval_dir+name, smoothed)

    fig.suptitle('Weather, Temp, PM25', fontsize=20)
    plt.savefig(st.eval_dir+name+'.png')
    plt.close()

def smooth_visualize_weather_test():
    print 'smoothing weather test data ...'
    name = 'weather_daywise_test'
    data = load(st.eval_dir_test+'weather'+'.bin')
    n_days = 31

    fig, axarr = plt.subplots(n_days, 3, figsize=(20, 30))
    plt.ioff()
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

    save(st.eval_dir_test+name, smoothed)

    fig.suptitle('Weather, Temp, PM25', fontsize=20)
    plt.savefig(st.eval_dir_test+name+'.png')
    plt.close()

# if __name__ == "__main__":
#     poi_correlation()
    # name = 'weather.bin'
    # weather_correlation(st.eval_dir+name)
    # smooth_weather_train()
    # smooth_weather_test()
    # visualize_weather(data, 'Weather', '(Weather, Temp, PM25)')