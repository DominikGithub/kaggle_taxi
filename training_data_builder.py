#!/usr/bin/env /home/dominik/anaconda2/bin/python

import math
import datetime
from utils_file import *
import statics as st

class Training_data_builder(object):

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
        # self.pois = load(st.eval_dir+'pois.bin')[:,:-15]

    def build_training_data_per_day(self):
        pass

    def build_training_data_per_week_day(self):
        pred_timeslots = [x.split('-')[3] for x in self.prediction_times]
        n_pred_tisl = len(pred_timeslots)

        sample_d_t = []
        gap_d_t = []
        for d in range(st.n_districts):
            for week_day in range(7):
                for dtime_slt in range(st.n_timeslots):
                    dem = self.demand_test[week_day, d, dtime_slt]
                    if math.isnan(dem): dem = self.demand_train[week_day, d, dtime_slt]
                    supp = self.supply_test[week_day, d, dtime_slt]
                    if math.isnan(supp):  supp = self.supply_train[week_day, d, dtime_slt]
                    params = [week_day, dtime_slt, dem, supp]
                    sample_d_t.append(np.concatenate((params,
                                                      self.traffic_train[week_day, d, dtime_slt, :].flatten(),
                                                      # self.pois[d].flatten(),
                                                      self.destination_train[week_day, d].flatten(),
                                                      self.start_train[week_day, d].flatten()
                                                      # ,
                                                      # self.weather_train[day, dtime_slt,:].flatten()
                                                      ), axis=0))
                    gap_d_t.append(self.gap_train[week_day, d, dtime_slt])

        sample_d_t_test = []
        gap_d_t_test = []
        for d in range(st.n_districts):
            for dtime_slt in range(n_pred_tisl):
                week_day = datetime.datetime.strptime(self.prediction_times[dtime_slt][:10], '%Y-%m-%d').weekday()
                if week_day == 0 or week_day == 2:
                    continue
                dem = self.demand_test[week_day, d, dtime_slt]
                if math.isnan(dem): dem = self.demand_train[week_day, d, dtime_slt]
                supp = self.supply_test[week_day, d, dtime_slt]
                if math.isnan(supp):  supp = self.supply_train[week_day, d, dtime_slt]
                params = [week_day, dtime_slt, dem, supp]
                sample_d_t_test.append(np.concatenate((params,
                                                       self.traffic_test[week_day, d, dtime_slt, :].flatten(),
                                                       # pois[d].flatten(),
                                                       self.destination_test[week_day, d].flatten(),
                                                       self.start_test[week_day, d].flatten()
                                                       # ,
                                                       # self.weather_test[day, dtime_slt,:].flatten()
                                                       ), axis=0))
                gap_d_t_test.append(self.gap_test[week_day, d, dtime_slt])
        return self.gap_train, sample_d_t, sample_d_t_test, gap_d_t, gap_d_t_test, self.prediction_times, n_pred_tisl

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