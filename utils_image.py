''' collection of utility methods for plotting results '''

import numpy as np
import datetime
import time
import statics as st
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from utils_date import norm
from utils_file import load
from correlation_smoothing import interpolate_traffic

def visualizations(interpolate_missing=False):
    visualize_orders(load(st.eval_dir+'demand.bin'), 'Demand', normalize=True)
    visualize_orders(load(st.eval_dir+'supply.bin'), 'Supply', normalize=True)
    visualize_orders(load(st.eval_dir+'gap.bin'), 'Gap', normalize=True)
    hist(load(st.eval_dir+'start_dist.bin'), 'Start_dist', y_range=[70, 180000])
    hist(load(st.eval_dir+'dest_dist.bin'), 'Dest_dist', y_range=[70, 120000])

    if interpolate_missing:
        traffic_data = interpolate_traffic(53)
        visualize_traffic(traffic_data, 'Traffic', normalize=True)
    else:
        visualize_traffic(load(st.eval_dir + 'traffic.bin'), 'Traffic', normalize=True)

    visualize_weather(load(st.data_dir + 'weather.bin'), 'Weather', '(Weather, Temp, PM25)')
    visualize_pois(load(st.eval_dir + 'pois.bin'), 'Pois level 1')
    visualize(load(st.eval_dir + 'pois_simple.bin'), 'Pois level 1_simpel')

def visualize(data, title, normalize=False):
    print 'plotting %s' % title

    if normalize: data = norm(data)

    plt.imshow(data.transpose(), interpolation='none', cmap=st.colormap, origin='lower', extent=[0, st.n_districts, 0, 24])
    axes = plt.gca()
    axes.set_xticks(xrange(0, st.n_districts+1, 11))
    axes.set_yticks(xrange(0, 25, 6))
    plt.colorbar()
    # axes.axis('off')
    plt.suptitle(title)

    plt.savefig(st.eval_dir+title+'.png')
    plt.close()

def visualize_pois(data, title):
    print 'plotting %s' % title

    fig = plt.figure(figsize=(30, 8))
    cols = st.n_poi_first // 2
    for fst in range(st.n_poi_first):
        for col in range(cols):
            ax2 = plt.subplot2grid((2, st.n_poi_first//2), (fst//cols, col))
            ax2.imshow(data[:,fst,:].transpose(), interpolation='none', cmap=st.colormap, origin='lower', extent=[0, st.n_districts, 0, st.n_poi_second])
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)

    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(st.eval_dir+title+'.png')
    plt.close()

def visualize_correlation(data, title):
    print 'plotting %s' % title

    plt.imshow(data.transpose(), interpolation='none', cmap='bwr')
    plt.colorbar()
    plt.suptitle(title)

    plt.savefig(title+'.png')
    plt.close()

def visualize_prediction(data, title, n_time_slots, timestmp, normalize=False):
    print 'plotting %s' % title

    if normalize: data = norm(data)

    plt.imshow(data.transpose(), interpolation='none', cmap=st.colormap, origin='lower', extent=[0, st.n_districts, 0, n_time_slots])
    axes = plt.gca()
    axes.set_xticks(xrange(0, st.n_districts+1, 11))
    # axes.set_yticks(xrange(0, n_time_slots, 6))
    axes.get_yaxis().set_visible(False)
    plt.colorbar()
    plt.suptitle(title)

    plt.savefig('predictions/'+title+'_'+str(timestmp)+'.png')
    plt.close()

def visualize_traffic(data, title, normalize=False):
    print 'plotting %s' % title
    fig = plt.figure(figsize=(30, 20))
    idx = 0
    for day in range(7):
        for lvl in range(st.max_congestion_lvls):
            ax2 = plt.subplot2grid((7, st.max_congestion_lvls), (day, lvl))
            if normalize:   dat = norm(data[:,:,idx])
            else:           dat = data[:,:,idx]
            ax2.imshow(dat[day].transpose(), interpolation='none', cmap=st.colormap, origin='lower', extent=[0, st.n_districts, 0, 24])
            axes = plt.gca()
            axes.set_xticks(xrange(0, st.n_districts + 1, 11))
            axes.set_yticks(xrange(0, 25, 6))
            plt.title(lvl)
            # plt.colorbar()
            idx += 1
    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(st.eval_dir+title+'.png')
    plt.close()

def visualize_orders(data, title, normalize=False):
    print 'plotting %s' % title
    fig = plt.figure(figsize=(5, 18))
    for day in range(7):
        ax2 = plt.subplot2grid((7, 1), (day, 0))
        if normalize:   dat = norm(data[day,:])
        else:           dat = data[day,:]
        ax2.imshow(dat.transpose(), interpolation='none', cmap=st.colormap, origin='lower', extent=[0, st.n_districts, 0, 24])
        axes = plt.gca()
        axes.set_xticks(xrange(0, st.n_districts + 1, 11))
        axes.set_yticks(xrange(0, 25, 6))
        # plt.title(lvl)
    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(st.eval_dir+title+'.png')
    plt.close()

def visualize_weather(data, name, title):
    print 'plotting %s' % title
    fig = plt.figure(figsize=(20, 30))
    for day in range(0, st.n_train_days):
        for obj in range(3):
            ax2 = plt.subplot2grid((st.n_train_days, 3), (day, obj))
            colors = ['red', 'green', 'blue']
            ax2.plot(data[day,:,obj], color=colors[obj])
            ax2.set_xticks(xrange(0, 144, 10))
            plt.xlim((0, st.n_timeslots))
    fig.suptitle(name+' '+title, fontsize=20)
    plt.tight_layout()
    plt.savefig(st.eval_dir+name+'.png')
    plt.close()

def hist(data, title, y_range=None):
    print 'plotting %s' % title
    fig = plt.figure(figsize=(40, 10))
    # n, bins, patches = plt.hist(data)

    error_config = {'ecolor': '0.3'}
    bar_width = 0.5
    for mode in range(2):
        for day in range(7):
            plt.subplot2grid((2,7), (mode, day))
            dat = data[day,:]
            index = np.arange(len(dat))
            if y_range and mode==1:
                plt.ylim(*y_range)
            plt.xlim(0, 66)
            axbar = plt.bar(index,
                            dat,
                            bar_width,
                            alpha=0.6,
                            color='g',
                            error_kw=error_config,
                            label='title'
            )

    plt.suptitle(title)
    plt.savefig(st.eval_dir+title+'.png')
    plt.close()

def visualize_oneDim(data, title):
    print 'plotting %s' % title
    fig = plt.figure()
    # ax = fig.gca()
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(np.mean(data, axis=0)[:,1:])
    # fig.autofmt_xdate()

    # weather = data[:,0]
    # x = range(len(weather))
    # plt.xticks(x, weather)

    temp_patch = mpatches.Patch(color='green', label='Temperature')
    pm25_patch = mpatches.Patch(color='blue', label='PM25 (air pollution)')
    weather_patch = mpatches.Patch(color='red', label='Weather')
    plt.legend(handles=[temp_patch, weather_patch, pm25_patch], prop={'size':6})

    # ax.set_xticks(xrange(1, 240, 20))
    plt.xlim((0, st.n_timeslots))
    plt.xlabel('time slots')
    # ax.get_yaxis().set_visible(False)
    plt.suptitle(title)

    plt.savefig(st.eval_dir+title+'.png')
    plt.close()
