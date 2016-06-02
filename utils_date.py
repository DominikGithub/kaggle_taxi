#!/usr/bin/env /home/dominik/anaconda2/bin/python

import numpy as np
import time
from datetime import datetime, timedelta
import statics as st

def get_timeslot_time(year, month, day, timeslot_idx):
    return time.mktime(datetime.strptime(('%s-%s-%s' % (year, month, day)), '%Y-%m-%d').timetuple())

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
    curr = get_timeslot_time(time_str[:4], time_str[6:7], time_str[9:10], 0)
    return int((curr - first_jan) / st.milli_sec_per_day)


def toUTCtimestamp(dt, epoch=datetime(1970, 1, 1)):
    td = dt - epoch
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10 ** 6) / 1e6

# def whiten(data):
#     print('... whiten input')
#     cov = T.matrix('cov', dtype=theano.config.floatX)
#     eig_vals, eig_vecs = tfunc([cov], T.nlinalg.eig(cov))(np.cov(data))
#     x = T.matrix('x')
#     vc = T.matrix('vc')
#     mat_inv = T.matrix('mat_inv')
#     np_eps = 0.1 * np.eye(eig_vals.shape[0])
#
#     sqr_inv = np.linalg.inv(np.sqrt(np.diag(eig_vals) + np_eps))
#     whitening = T.dot(T.dot(T.dot(vc, mat_inv), vc.T), x)
#     zca = tfunc([x, vc, mat_inv], whitening, allow_input_downcast=True)
#     return zca(data, eig_vecs, sqr_inv)

def norm(data, axis=1):
    frac = np.zeros_like(data)
    row_sums = data.sum(axis=axis, keepdims=True)
    frac = data / row_sums
    return np.nan_to_num(frac)