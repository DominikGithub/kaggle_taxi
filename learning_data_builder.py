#!/usr/bin/env /home/dominik/anaconda2/bin/python

import numpy as np
import math
from datetime import datetime
import theano
import theano.tensor as T
from theano import function as tfunc
from utils_file import *
import statics as st

class Learning_data_builder(object):

    def __init__(self, logger, interpolate_missing=False):
        self.logger = logger
        self.empty_train_days = [3,7,11, 15, 19, 24]
        self.empty_test_days = [23, 25, 27, 29]
        self.load_prediction_times()
        # self.traffic = load(st.eval_dir_test+'traffic.bin')
        # if interpolate_missing:
        #     self.traffic = interpolate_traffic(53)
        # else:
        #     self.traffic = load(st.eval_dir_test+'traffic.bin')
        # self.traffic_train = np.asarray([norm(jam) for jam in load(st.eval_dir+'traffic.bin')])

    def load_prediction_times(self):
        self.prediction_times = []
        with open(st.data_dir_test + 'read_me_1.txt') as f:
            prediction_date = f.read().splitlines()
        for p in prediction_date:
            self.prediction_times.append(p)
        self.prediction_times = self.prediction_times[st.n_csv_header_lines:]
        self.pred_days = [x.split('-')[2] for x in self.prediction_times]
        self.pred_timeslots = [dict(zip(st.prediction_keys, x.split('-'))) for x in self.prediction_times]
        self.n_pred_tisl = len(self.pred_timeslots)

    def normalize(self, data):
        eps_norm = 1    #10 proposed for value range of 255
        print('... normalize input eps_norm: %s ' % eps_norm)
        self.logger.info('... normalizing input (eps_norm: %s) ' % eps_norm)
        data = np.asarray(data)
        self.means = data.mean(axis=1, keepdims=True)
        self.x_mean = data - self.means
        self.variance = np.var(data, axis=1, keepdims=False)
        return self.x_mean / (np.sqrt(self.variance + eps_norm))[:, np.newaxis]

    def whiten(self, data):
        eps = 0.1
        print('... whiten input (eps: %s)' % eps)
        self.logger.info('... whiten input (eps: %s)' % eps)
        cov = T.matrix('cov', dtype=theano.config.floatX)
        eig_vals, eig_vecs = tfunc([cov], T.nlinalg.eig(cov), allow_input_downcast=True)(np.cov(data))
        x = T.matrix('x')
        vc = T.matrix('vc')
        mat_inv = T.matrix('mat_inv')
        np_eps = eps * np.eye(eig_vals.shape[0])

        sqr_inv = np.linalg.inv(np.sqrt(np.diag(eig_vals) + np_eps))
        whitening = T.dot(T.dot(T.dot(vc, mat_inv), vc.T), x)
        zca = tfunc([x, vc, mat_inv], whitening, allow_input_downcast=True)
        return zca(data, eig_vecs, sqr_inv)

    def load_week_day_wise_data(self):
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

    def load_daywise_data(self):
        self.load_prediction_times()
        self.demand_train = load(st.eval_dir + 'demand_daywise.bin')
        self.demand_test = load(st.eval_dir_test + 'demand_daywise.bin')

        self.supply_train = load(st.eval_dir + 'supply_daywise.bin')
        self.supply_test = load(st.eval_dir_test + 'supply_daywise.bin')

        self.gap_train = load(st.eval_dir + 'gap_daywise.bin')
        self.gap_test = load(st.eval_dir_test + 'gap_daywise.bin')

        self.start_train = load(st.eval_dir + 'start_dist_daywise.bin')
        self.start_test = load(st.eval_dir_test + 'start_dist_daywise.bin')

        self.dest_train = load(st.eval_dir + 'dest_dist_daywise.bin')
        self.dest_test = load(st.eval_dir_test + 'dest_dist_daywise.bin')

        self.traffic_train = load(st.eval_dir + 'traffic_daywise.bin')
        self.traffic_test = load(st.eval_dir_test + 'traffic_daywise.bin')

        self.weather_train = load(st.eval_dir + 'weather_daywise.bin')
        self.weather_test = load(st.eval_dir_test + 'weather_daywise_test.bin')

        self.pois = load(st.eval_dir_test + 'pois.bin')
        # self.pois = load(st.eval_dir_test + 'pois_simple.bin')

    def build_daily_training_data(self):
        self.logger.info('... building training data: daily samples')
        self.load_daywise_data()
        samples_train = []
        gap_train_rslt = []

        for distr in range(st.n_districts):
            for dtime_slt in range(st.n_timeslots):
                for day in range(len(self.gap_train)):
                    skip_day = True
                    try:
                        self.empty_train_days.index(day)
                    except:
                        skip_day = False
                    if skip_day or day > 30:
                        continue

                    samples_train.append(np.concatenate(([day, dtime_slt],
                                                         self.traffic_train[day, distr, dtime_slt, :].flatten(),
                                                         # self.pois[distr].flatten(),
                                                         self.dest_train[day, distr].flatten(),
                                                         self.start_train[day, distr].flatten(),
                                                         self.demand_train[day, distr, dtime_slt].flatten(),
                                                         self.supply_train[day, distr, dtime_slt].flatten(),
                                                         self.weather_train[day, :, dtime_slt].flatten()
                                                     ), axis=0))
                    gap_train_rslt.append(self.gap_train[day, distr, dtime_slt])

        samples_test = []
        gap_test_rslt = []
        for distr in range(st.n_districts):
            for pred_idx, pred_dict in enumerate(self.pred_timeslots):
                day = int(pred_dict.get('day'))
                dtime_slt = int(pred_dict.get('timeslot'))
                skip_day = True
                try:
                    self.empty_test_days.index(day)
                except:
                    skip_day = False
                if skip_day or day > 30:
                    continue

                samples_test.append(np.concatenate(([day, dtime_slt],
                                                     self.traffic_test[day, distr, dtime_slt, :].flatten(),
                                                     # self.pois[distr].flatten(),
                                                     self.dest_test[day, distr].flatten(),
                                                     self.start_test[day, distr].flatten(),
                                                     self.demand_test[day, distr, dtime_slt].flatten(),
                                                     self.supply_test[day, distr, dtime_slt].flatten(),
                                                     self.weather_test[day, :, dtime_slt].flatten()
                                                     ), axis=0))
                gap_test_rslt.append(self.gap_test[day, distr, dtime_slt])

        n_train = len(samples_train)
        samples_all = np.concatenate((samples_train, samples_test), axis=0)
        samples_all = self.normalize(samples_all).transpose()
        samples_all = self.whiten(samples_all).transpose()
        samples_train = samples_all[:n_train]
        samples_test = samples_all[n_train:]

        # samples_train = self.normalize(samples_train).transpose()
        # samples_train = self.whiten(samples_train).transpose()
        # samples_test = self.normalize(samples_test).transpose()
        # samples_test = self.whiten(samples_test).transpose()

        # samples_train, samples_test = self.normalize(samples_train, samples_test).transpose()
        # samples_train, samples_test = self.whiten(samples_train, samples_test).transpose()

        return samples_train, samples_test, gap_train_rslt, gap_test_rslt, self.prediction_times, self.n_pred_tisl

    def build_training_data_per_week_day(self):
        self.logger.info('... building training data: samples per week day')
        self.load_week_day_wise_data()
        pred_timeslots = [x.split('-')[3] for x in self.prediction_times]
        n_pred_tisl = len(pred_timeslots)

        samples_train = []
        gap_train_rslt = []
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
                    gap_train_rslt.append(self.gap_train[week_day, d, dtime_slt])

        samples_test = []
        gap_test_rslt = []
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
                gap_test_rslt.append(self.gap_test[week_day, d, dtime_slt])

        # n_train = len(samples_train)
        # samples_all = np.concatenate((samples_train, samples_test), axis=0)
        # samples_all = self.normalize(samples_all).transpose()
        # samples_all = self.whiten(samples_all).transpose()
        #
        # samples_train = samples_all[:n_train]
        # samples_test = samples_all[n_train:]

        samples_train = self.normalize(samples_train).transpose()
        samples_train = self.whiten(samples_train).transpose()
        samples_test = self.normalize(samples_test).transpose()
        samples_test = self.whiten(samples_test).transpose()

        return self.gap_train, samples_train, samples_test, gap_train_rslt, gap_test_rslt, self.prediction_times, n_pred_tisl