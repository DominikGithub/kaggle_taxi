#!/usr/bin/env /home/dominik/anaconda2/bin/python

import os
import numpy as np
import csv
import glob
import cPickle
import theano
import itertools
import six.moves.cPickle as pickle
import statics as st
from datetime import datetime
from utils_date import toUTCtimestamp

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
    pickle.dump(data, output, protocol=cPickle.HIGHEST_PROTOCOL)
    output.close()

def save_model(logging, classifier):
    timestmp = str(toUTCtimestamp(datetime.utcnow()))
    dot_idx = timestmp.index('.')
    print '.... save model at: %s' % timestmp
    logging.info('.... save model at: %s' % timestmp)

    save(st.model_dir+'model_params_'+timestmp[:dot_idx], [param.get_value() for param in classifier.params])

    # save(st.model_dir+'model_h1W_'+timestmp, classifier.hiddenLayer1.W.get_value(borrow=True))
    # save(st.model_dir+'model_h1b_'+timestmp, classifier.hiddenLayer1.b.get_value(borrow=True))
    # try:
    #     save(st.model_dir+'model_h2W_'+timestmp, classifier.hiddenLayer2.W.get_value(borrow=True))
    #     save(st.model_dir+'model_h2b_'+timestmp, classifier.hiddenLayer2.b.get_value(borrow=True))
    # except AttributeError as ex:
    #     pass
    # save(st.model_dir+'model_oW_'+timestmp, classifier.outputLayer.W.get_value(borrow=True))
    # save(st.model_dir+'model_ob_'+timestmp, classifier.outputLayer.b.get_value(borrow=True))
    # os.system('espeak "your model has been saved"')

def load_model(classifier):
    latest_timestamp = -np.inf
    file_list = glob.glob(st.model_dir+'model_*')
    if len(file_list) == 0:
        raise ImportWarning('No saved model found')
    for file in file_list:
        timestamp = int(file[-14:-4])
        if timestamp > latest_timestamp:
            latest_timestamp = timestamp
    latest_timestamp = str(latest_timestamp)
    for file in file_list:
        file_name = file[7:-4]
        # if file_name == 'model_h1W_'+latest_timestamp:
        #     param_file = open(file)
        #     classifier.hiddenLayer1.W.set_value(cPickle.load(param_file), borrow=True)
        # if file_name == 'model_h1b_' + latest_timestamp:
        #     param_file = open(file)
        #     classifier.hiddenLayer1.b.set_value(cPickle.load(param_file), borrow=True)

        # if file_name == 'model_h2W_' + latest_timestamp and classifier.hiddenLayer2:
        #     param_file = open(file)
        #     classifier.hiddenLayer2.W.set_value(cPickle.load(param_file), borrow=True)
        # if file_name == 'model_h2b_' + latest_timestamp and classifier.hiddenLayer2:
        #     param_file = open(file)
        #     classifier.hiddenLayer2.b.set_value(cPickle.load(param_file), borrow=True)

        # if file_name == 'model_oW_' + latest_timestamp:
        #     param_file = open(file)
        #     classifier.outputLayer.W.set_value(cPickle.load(param_file), borrow=True)
        # if file_name == 'model_ob_' + latest_timestamp:
        #     param_file = open(file)
        #     classifier.outputLayer.b.set_value(cPickle.load(param_file), borrow=True)

        if file_name == 'model_params_' + latest_timestamp:
            param_file = open(file)
            classifier.params = [theano.shared(param, borrow=True) for param in cPickle.load(param_file)]

            classifier.hiddenLayer1.W.set_value(classifier.params[0].get_value(borrow=True), borrow=True)
            classifier.hiddenLayer1.b.set_value(classifier.params[1].get_value(borrow=True), borrow=True)

            if len(classifier.params) > 4:
                classifier.hiddenLayer2.W.set_value(classifier.params[2].get_value(borrow=True), borrow=True)
                classifier.hiddenLayer2.b.set_value(classifier.params[3].get_value(borrow=True), borrow=True)
                classifier.outputLayer.W.set_value(classifier.params[4].get_value(borrow=True), borrow=True)
                classifier.outputLayer.b.set_value(classifier.params[5].get_value(borrow=True), borrow=True)
            else:
                classifier.outputLayer.W.set_value(classifier.params[2].get_value(borrow=True), borrow=True)
                classifier.outputLayer.b.set_value(classifier.params[3].get_value(borrow=True), borrow=True)

    return latest_timestamp, classifier

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