#!/usr/bin/env /home/dominik/anaconda2/bin/python

import os
import numpy as np
import csv
import glob
import itertools
import six.moves.cPickle as pickle
import statics as st
import matplotlib.pyplot as plt

def load_csv(filename, limit=None):
    filename = os.path.join(st.fileDir, filename)
    filename = os.path.abspath(os.path.realpath(filename))

    lines = []
    f = open(filename, 'rb')
    try:
        reader = csv.reader(f, delimiter='\t')
        for row in itertools.islice(reader, limit):
            # print row
            lines.append(row)
    finally:
        f.close()
    return lines

def merge_files(order_files):
    orders = []
    for o in order_files:
        # print o
        orders += load_csv(o)
    return orders

def load_test_data():
    date = '*'
    traffic_jam = merge_files(glob.glob(st.data_dir + 'traffic_data_'+date))

# def hist(x):
#     colors = ['red']    #, 'tan', 'lime'
#     n, bins, patches = plt.hist(x, 144, normed=False)#, color=colors
#     plt.xlabel('thing')
#     plt.ylabel('count')
#     # plt.axis([0, 15, 0, 1])
#     plt.grid(True)
#     plt.show()

def load(filename):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, filename)
    filename = os.path.abspath(os.path.realpath(filename))
    fo = open(filename, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def save(filename, data):
    output = open(filename+'.bin', 'wb')
    pickle.dump(data, output)
    output.close()

def save_predictions(timeslots, predictions, timestmp):
    with open('predictions/predictions_'+str(timestmp)+'.csv', 'wb') as f:
        writer = csv.writer(f, delimiter=',')

        csv_data = ['District ID', 'Time slot', 'Prediction value']
        for t in timeslots:
            for d in range(st.n_districts):
                t_idx = timeslots.index(t)
                val = predictions[d, t_idx]
                csv_data = np.vstack([csv_data, [d+1, t, val]])
        writer.writerows(csv_data)