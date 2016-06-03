#!/usr/bin/env /home/dominik/anaconda2/bin/python

import numpy as np
import math
import datetime
import theano
import theano.tensor as T
from theano import function as tfunc
from utils_file import *
import statics as st

class Learning_data_builder(object):

    def __init__(self, interpolate_missing=False):
        self.prediction_times = []
        with open(st.data_dir_test + 'read_me_1.txt') as f:
            prediction_date = f.read().splitlines()
        for p in prediction_date:
            self.prediction_times.append(p)
        self.prediction_times = self.prediction_times[st.n_csv_header_lines:]
        # self.traffic = load(st.eval_dir_test+'traffic.bin')
        # if interpolate_missing:
        #     self.traffic = interpolate_traffic(53)
        # else:
        #     self.traffic = load(st.eval_dir_test+'traffic.bin')
        # self.traffic_train = np.asarray([norm(jam) for jam in load(st.eval_dir+'traffic.bin')])
        self.traffic_train = load(st.eval_dir + 'traffic.bin')
        self.traffic_test = load(st.eval_dir_test + 'traffic.bin')

        self.destination_train = load(st.eval_dir + 'dest_dist.bin')
        self.destination_test = load(st.eval_dir_test + 'dest_dist.bin')
        self.start_train = load(st.eval_dir_test + 'start_dist.bin')
        self.start_test = load(st.eval_dir + 'start_dist.bin')
        # self.weather_train = load(st.eval_dir+'weather.bin')
        # self.weather_test = load(st.eval_dir_test+'weather.bin')

        self.demand_test = load(st.eval_dir_test + 'demand.bin')
        self.demand_train = load(st.eval_dir + 'demand.bin')
        self.supply_test = load(st.eval_dir_test + 'supply.bin')
        self.supply_train = load(st.eval_dir + 'supply.bin')
        self.gap_train = load(st.eval_dir + 'gap.bin')
        self.gap_test = load(st.eval_dir_test + 'gap.bin')
        # self.pois = load(st.eval_dir+'pois.bin')
        self.pois = load(st.eval_dir + 'pois_simple.bin')

    def normalize(self, data):
        eps_norm = 1
        print('... normalize input (calculating statistics)')
        data = np.asarray(data)
        # if not hasattr(self, 'means'):
        self.means = data.mean(axis=1, keepdims=True)
        # else:
        #     print '... normalizing using precalculated statistics'
        # if not hasattr(self, 'x_mean'):
        self.x_mean = data - self.means
        # if not hasattr(self, 'variance'):
        self.variance = np.var(data, axis=1, keepdims=False)
        return self.x_mean / (np.sqrt(self.variance + eps_norm))[:, np.newaxis]

    def whiten(self, data):
        print('... whiten input')
        cov = T.matrix('cov', dtype=theano.config.floatX)
        eig_vals, eig_vecs = tfunc([cov], T.nlinalg.eig(cov), allow_input_downcast=True)(np.cov(data))
        x = T.matrix('x')
        vc = T.matrix('vc')
        mat_inv = T.matrix('mat_inv')
        np_eps = 0.1 * np.eye(eig_vals.shape[0])

        sqr_inv = np.linalg.inv(np.sqrt(np.diag(eig_vals) + np_eps))
        whitening = T.dot(T.dot(T.dot(vc, mat_inv), vc.T), x)
        zca = tfunc([x, vc, mat_inv], whitening, allow_input_downcast=True)
        return zca(data, eig_vecs, sqr_inv)

    def build_training_data_per_day(self):
        pred_timeslots = [x.split('-')[3] for x in self.prediction_times]
        n_pred_tisl = len(pred_timeslots)

        samples_train = []
        gap_train = []
        # for d in range(st.n_districts):
        #     for week_day in range(7):


        samples_test = []
        gap_test = []
        # for d in range(st.n_districts):
        #     for dtime_slt in range(n_pred_tisl):

        # samples_train = self.normalize(samples_train).transpose()
        # samples_train = self.whiten(samples_train).transpose()
        # samples_test = self.normalize(samples_test).transpose()
        # samples_test = self.whiten(samples_test).transpose()
        # return self.gap_train, samples_train, samples_test, gap_train, gap_test, self.prediction_times, n_pred_tisl

    def build_training_data_per_week_day(self):
        pred_timeslots = [x.split('-')[3] for x in self.prediction_times]
        n_pred_tisl = len(pred_timeslots)

        samples_train = []
        gap_train = []
        for d in range(st.n_districts):
            for week_day in range(7):
                for dtime_slt in range(st.n_timeslots):
                    dem = self.demand_test[week_day, d, dtime_slt]
                    if math.isnan(dem): dem = self.demand_train[week_day, d, dtime_slt]
                    supp = self.supply_test[week_day, d, dtime_slt]
                    if math.isnan(supp):  supp = self.supply_train[week_day, d, dtime_slt]
                    params = [week_day, dtime_slt, dem, supp]
                    samples_train.append(np.concatenate((params,
                                                        self.traffic_train[week_day, d, dtime_slt, :].flatten(),
                                                        self.pois[d].flatten(),
                                                        self.destination_train[week_day, d].flatten(),
                                                        self.start_train[week_day, d].flatten()
                                                        # ,
                                                        # self.weather_train[day, dtime_slt,:].flatten()
                                                        ), axis=0))
                    gap_train.append(self.gap_train[week_day, d, dtime_slt])

        samples_test = []
        gap_test = []
        for d in range(st.n_districts):
            for dtime_slt in range(n_pred_tisl):
                week_day = datetime.datetime.strptime(self.prediction_times[dtime_slt][:10], '%Y-%m-%d').weekday()
                dem = self.demand_test[week_day, d, dtime_slt]
                if math.isnan(dem): dem = self.demand_train[week_day, d, dtime_slt]
                supp = self.supply_test[week_day, d, dtime_slt]
                if math.isnan(supp):  supp = self.supply_train[week_day, d, dtime_slt]
                params = [week_day, dtime_slt, dem, supp]
                samples_test.append(np.concatenate((params,
                                                    self.traffic_test[week_day, d, dtime_slt, :].flatten(),
                                                    self.pois[d].flatten(),
                                                    self.destination_test[week_day, d].flatten(),
                                                    self.start_test[week_day, d].flatten()
                                                    # ,
                                                    # self.weather_test[day, dtime_slt,:].flatten()
                                                    ), axis=0))
                gap_test.append(self.gap_test[week_day, d, dtime_slt])

        samples_train = self.normalize(samples_train).transpose()
        samples_train = self.whiten(samples_train).transpose()

        samples_test = self.normalize(samples_test).transpose()
        samples_test = self.whiten(samples_test).transpose()

        return self.gap_train, samples_train, samples_test, gap_train, gap_test, self.prediction_times, n_pred_tisl

    # def load_pred_times(self):
    #     prediction_times = []
    #     with open(st.data_dir_test + 'read_me_1.txt') as f:
    #         prediction_date = f.read().splitlines()
    #     for p in prediction_date:
    #         prediction_times.append(p)
    #     prediction_times = prediction_times[st.n_csv_header_lines:]
    #     pred_timeslots = [x.split('-')[3] for x in prediction_times]
    #     return pred_timeslots, prediction_times
    #
    #     pred_timeslots, prediction_times = load_pred_times()
    #     n_pred_tisl = len(pred_timeslots)
    #
    #     pred_timeslots, prediction_times = load_pred_times()
    #     n_pred_tisl = len(pred_timeslots)
    #     return prediction_times, n_pred_tisl