#!/usr/bin/env /home/dominik/anaconda2/bin/python

import os

fileDir = os.path.dirname(os.path.realpath('__file__'))

data_dir_test = 'data_test/'

cut_off_test = 5
data_dir = 'data_train/'#data_dir_test

eval_dir = data_dir+'evaluation/'
eval_dir_test = data_dir_test+'evaluation/'

# n_train_days = 21   #5
n_districts = 66   # 66 kown disctricts + unkonwn many unkonwn
n_timeslots = 144

driver_id = 'driver_id'
order_id = 'order_id'
passenger_id = 'passenger_id'
start_district_hash = 'start_district_hash'
dest_district_hash = 'dest_district_hash'
price = 'price'
time = 'time'

tj_district_hash = 'tj_district_hash'
tj_level = 'tj_level'
max_congestion_lvls = 4

w_weather = 'weather'
w_pm25 = 'pm25'
w_temperature = 'temperature'

order_keys = [order_id, driver_id, passenger_id, start_district_hash, dest_district_hash, price, time]
weather_keys = [time, w_weather, w_temperature, w_pm25]
prediction_keys = ['year', 'month', 'day', 'timeslot']

n_poi_first = 26
n_poi_second = 19
n_poi_third = 2

milli_sec_per_day = 64800

n_csv_header_lines = 1
colormap='plasma'