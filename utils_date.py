#!/usr/bin/env /home/dominik/anaconda2/bin/python

import numpy as np
import time
from datetime import datetime
import statics as st

def get_timeslot_time(year, month, day, timeslot_idx):
    str = '%s-%s-%s' % (year, month.zfill(2), day.zfill(2))
    return time.mktime(datetime.strptime(str, '%Y-%m-%d').timetuple())

def get_timeslot(time_point):
    '''
    get the timeslot index, one day divided into 144 timeslots, each 10 Min long
    '''
    year, month, day = time_point[:10].split('-')
    morning = get_timeslot_time(year, month, day, 0)
    curr = toTimeStmp(time_point)
    return int(np.floor((curr - morning) / 10/60))

def toTimeStmp(time_str):
    return time.mktime(datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timetuple())

def get_day(time_str):
    first_jan = get_timeslot_time('2016', '01', '01', 0)
    curr = get_timeslot_time(time_str[:4], time_str[5:7], time_str[8:10], 0)
    return int((curr - first_jan) / st.milli_sec_per_day)

def toUTCtimestamp(dt, epoch=datetime(1970, 1, 1)):
    td = dt - epoch
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10 ** 6) / 1e6
