#!/usr/bin/env /home/dominik/anaconda2/bin/python

from sys import stdout
import threading
from utils_file import *
from utils_image import *
from utils_date import *
from data_container import *
from correlation_smoothing import *

class Preprocessor(object):
    file_postfix = '_daywise'

    def __init__(self, interpolate_missing=False):
        # self.process_orders_day_wise(date='2016-01-*')
        # self.process_traffic_day_wise(date='2016-01-*', interpolate_missing=False)
        # self.preprocess_weather(date='2016-01-*')

        # visualize_orders(self.normalize(load(st.eval_dir + 'demand'+self.file_postfix+'.bin')), 'Demand'+self.file_postfix)
        # visualize_orders(self.normalize(load(st.eval_dir + 'supply'+self.file_postfix+'.bin')), 'Supply'+self.file_postfix)
        # visualize_orders(self.normalize(load(st.eval_dir + 'gap'+self.file_postfix+'.bin')), 'Gap'+self.file_postfix)
        # hist(load(st.eval_dir + 'start_dist'+self.file_postfix+'.bin'), 'Start_dist'+self.file_postfix, y_range=[70, 180000])
        # hist(load(st.eval_dir + 'dest_dist'+self.file_postfix+'.bin'), 'Dest_dist'+self.file_postfix, y_range=[70, 120000])
        # visualize_traffic(self.normalize(load(st.eval_dir + 'traffic'+self.file_postfix+'.bin')), 'Traffic'+self.file_postfix)
        os.system('espeak "Preprocessing has finished"')

    def normalize(self, data):
        eps_norm = 1
        print('... normalize input (calculating statistics)')
        data = np.asarray(data)
        self.means = data.mean(axis=1, keepdims=True)
        self.x_mean = data - self.means
        self.variance = np.var(data, axis=1, keepdims=False)
        return self.x_mean / (np.sqrt(self.variance + eps_norm))[:, np.newaxis]

    def preprocess_weather(self, date):
        print 'preprocessing weather'
        train_case = {'days': 32, 'tistmp_day_start': -2, 'tistmp_day_end': None, 'path_data': st.data_dir,       'path_eval': st.eval_dir}
        test_case =  {'days': 32, 'tistmp_day_start': -7, 'tistmp_day_end': -5,     'path_data': st.data_dir_test,'path_eval': st.eval_dir_test}
        cases = [train_case, test_case]

        for case in cases:
            weather_mat = np.zeros(shape=(case.get('days'), int(st.n_timeslots), 3))
            for w_file in glob.glob(case.get('path_data') + 'weather_data_' + date):
                day_idx = int(w_file[case.get('tistmp_day_start') : case.get('tistmp_day_end')])
                # day_idx = int(w_file[-7:-5])    # test data
                # day_idx = int(w_file[-2:])    # train data
                w_data = load_csv(w_file)

                # if st.data_dir == 'data_train/':
                #     week_day = datetime.strptime(w_file[-10:], '%Y-%m-%d').weekday()
                # else:
                #     week_day = datetime.strptime(w_file[-10-st.cut_off_test:-st.cut_off_test], '%Y-%m-%d').weekday()

                for sing_w in w_data:
                    w_dict = dict(zip(st.weather_keys, sing_w))
                    tslot_idx = int(get_timeslot(w_dict[st.time]))
                    w_temp = w_dict[st.w_temperature]
                    w_pm25 = w_dict[st.w_pm25]
                    w_weather = w_dict[st.w_weather]

                    weather_mat[day_idx - 1][tslot_idx] = [w_weather, w_temp, w_pm25]

            save(case.get('path_eval') + 'weather', weather_mat)
        smooth_visualize_weather_train()
        smooth_visualize_weather_test()

    def process_traffic_day_wise(self, date, interpolate_missing=False):
        print 'preprocessing traffic'
        file_list = glob.glob(st.data_dir + 'traffic_data_' + date)
        n_days = len(file_list)
        for file_name in file_list:
            day = get_day(file_name[-10:])  # set 15 for test data
            if n_days < day:
                n_days = day
        n_days += 1
        traff_map = np.zeros(shape=(n_days, int(st.n_districts), int(st.n_timeslots), int(st.max_congestion_lvls)))
        dist_map = create_distr_Map()

        for traffic_file in file_list:
            traffic_data = load_csv(traffic_file)

            # if st.data_dir == 'data_train/':
            #     week_day = datetime.strptime(traffic_file[-10:], '%Y-%m-%d').weekday()
            # else:
            #     week_day = datetime.strptime(traffic_file[-10 - st.cut_off_test:-st.cut_off_test],
            #                                  '%Y-%m-%d').weekday()

            for sing_traf in traffic_data:
                distr_hash = sing_traf[0]
                distr_idx = int(dist_map.getkey_or_create(distr_hash)) - 1
                tj_lvl = sing_traf[1:len(sing_traf) - 1]
                time_str = sing_traf[len(sing_traf) - 1]
                day = get_day(time_str)
                tslot_idx = int(get_timeslot(time_str))

                for lvl in range(len(tj_lvl)):
                    traff_map[day][distr_idx][tslot_idx][lvl] = tj_lvl[lvl].split(':')[1]

        if interpolate_missing:
            save(st.eval_dir + 'traffic', traff_map)
            traff_map = interpolate_traffic(53)

        save(st.eval_dir + 'traffic'+self.file_postfix, traff_map)

    def process_orders_day_wise(self, date):
        print 'preprocessing orders'
        dist_map = create_distr_Map()
        driver_map = IncMap()
        passenger_map = IncMap()

        file_list = glob.glob(st.data_dir + 'order_data_' + date)
        n_days = len(file_list)
        for file_name in file_list:
            day = get_day(file_name[-10:])  # set 15 for test data
            if n_days < day:
                n_days = day
        n_days += 1
        orders = merge_files(file_list)

        supply      = np.zeros(shape=(n_days, st.n_districts, st.n_timeslots))
        demand      = np.zeros(shape=(n_days, st.n_districts, st.n_timeslots))
        gap         = np.zeros(shape=(n_days, st.n_districts, st.n_timeslots))
        start_dist  = np.zeros(shape=(n_days, st.n_districts))
        dest_dist   = np.zeros(shape=(n_days, st.n_districts))

        n_orders = len(orders)
        for counter, sing_ord in enumerate(orders):
            if np.mod(counter, np.floor(n_orders / 100)) == 0:
                stdout.write('\r%.2f%%' % (float(counter * 100) / n_orders))
                stdout.flush()

            order = dict(zip(st.order_keys, sing_ord))

            driver_map.incrementAt(order[st.driver_id])
            passenger_map.incrementAt(order[st.passenger_id])

            s_distr_idx = int(dist_map.getkey_or_create(order[st.start_district_hash])) - 1
            d_distr_idx = int(dist_map.getkey_or_create(order[st.dest_district_hash])) - 1
            # week_day = datetime.strptime(order[st.time], '%Y-%m-%d %H:%M:%S').weekday()
            day = get_day(order[st.time])
            tslot_idx = int(get_timeslot(order[st.time]))

            if order[st.driver_id] == 'NULL':
                gap[day][s_distr_idx][tslot_idx] += 1
            else:
                supply[day][s_distr_idx][tslot_idx] += 1
            demand[day][s_distr_idx][tslot_idx] += 1
            start_dist[day][s_distr_idx] += 1
            if d_distr_idx < st.n_districts:
                dest_dist[day][d_distr_idx] += 1

        print '\r'
        save(st.eval_dir + 'gap'+self.file_postfix, gap)
        save(st.eval_dir + 'demand'+self.file_postfix, demand)
        save(st.eval_dir + 'supply'+self.file_postfix, supply)

        save(st.eval_dir + 'start_dist'+self.file_postfix, start_dist)
        save(st.eval_dir + 'dest_dist'+self.file_postfix, dest_dist)
