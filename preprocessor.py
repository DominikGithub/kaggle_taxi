#!/usr/bin/env /home/dominik/anaconda2/bin/python

from sys import stdout
import threading
from utils_file import *
from utils_image import *
from utils_date import *
from data_container import *

class Preprocessor(object):
    file_postfix = '_daywise'

    def __init__(self, interpolate_missing=False):
        self.process_orders_day_wise(date='2016-01-*')

        print 'visualizing data'
        visualize_orders(load(st.eval_dir + 'demand'+self.file_postfix+'.bin'), 'Demand'+self.file_postfix, normalize=True)
        visualize_orders(load(st.eval_dir + 'supply'+self.file_postfix+'.bin'), 'Supply'+self.file_postfix, normalize=True)
        visualize_orders(load(st.eval_dir + 'gap'+self.file_postfix+'.bin'), 'Gap'+self.file_postfix, normalize=True)
        hist(load(st.eval_dir + 'start_dist'+self.file_postfix+'.bin'), 'Start_dist'+self.file_postfix, y_range=[70, 180000])
        hist(load(st.eval_dir + 'dest_dist'+self.file_postfix+'.bin'), 'Dest_dist'+self.file_postfix, y_range=[70, 120000])
        os.system('espeak "Preprocessing has finished"')

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
