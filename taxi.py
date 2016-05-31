#!/usr/bin/env /home/dominik/anaconda2/bin/python

from sys import stdout
import logging
import glob
import theano
from sklearn import linear_model
import math
from mlp import mlp_train
from data_container import HashMap, IncMap, create_distr_Map
from utils_file import merge_files, load, save, load_csv, load_test_data, save_predictions
from utils_image import *
from utils_data import norm, toUTCtimestamp, get_timeslot
from correlation_smoothing import smooth_weather, smooth_weather_test

logging.basicConfig(filename='taxi.log', level=logging.INFO)

def interpolate_traffic(district_idx):
    week = load(st.eval_dir + 'traffic.bin')
    all_days = np.ndarray((7, st.n_districts, 144, 4))
    idx = 0
    for day in week:
        clean_data = np.delete(day, district_idx, axis=0)
        mean_traffic = np.sum(clean_data, axis=0, keepdims=True) / len(clean_data)
        first = day[:district_idx,:,:]
        second = day[district_idx+1:,:,:]
        all_days[idx] = np.concatenate((first, mean_traffic, second), axis=0)
        idx += 1

    return all_days

def preprocess_traffic(date, interpolate_missing=False):
    print 'preprocessing traffic'
    dist_map = create_distr_Map()
    traff_map = np.zeros(shape=(7, int(st.n_districts), int(st.n_timeslots), int(st.max_congestion_lvls)))

    for traffic_file in glob.glob(st.data_dir+'traffic_data_'+date):
        traffic_data = load_csv(traffic_file)

        if st.data_dir == 'data/':
            week_day = datetime.datetime.strptime(traffic_file[-10:], '%Y-%m-%d').weekday()
        else:
            week_day = datetime.datetime.strptime(traffic_file[-10-st.cut_off_test:-st.cut_off_test], '%Y-%m-%d').weekday()

        for sing_traf in traffic_data:
            distr_hash = sing_traf[0]
            distr_idx = int(dist_map.getkey_or_create(distr_hash))-1
            tj_lvl = sing_traf[1:len(sing_traf)-1]
            tslot_idx =  int(get_timeslot(sing_traf[len(sing_traf)-1]))-1

            for lvl in range(len(tj_lvl)):
                traff_map[week_day][distr_idx][tslot_idx][lvl] = tj_lvl[lvl].split(':')[1]

    if interpolate_missing:
        save(st.eval_dir + 'traffic', traff_map)
        traff_map = interpolate_traffic(53)

    save(st.eval_dir+'traffic', traff_map)

def preprocess_orders(date):
    print 'preprocessing orders'
    dist_map = create_distr_Map()
    driver_map = IncMap()
    passenger_map = IncMap()
    supply = np.zeros(shape=(7, int(st.n_districts), int(st.n_timeslots)))
    demand = np.zeros(shape=(7, int(st.n_districts), int(st.n_timeslots)))
    gap = np.zeros(shape=(7, int(st.n_districts), int(st.n_timeslots)))
    start_dist = np.zeros(shape=(7, int(st.n_districts)))
    dest_dist = np.zeros(shape=(7, st.n_districts))

    orders = merge_files(glob.glob(st.data_dir+'order_data_'+date))
    n_orders = len(orders)
    for counter, sing_ord in enumerate(orders):
        if np.mod(counter, np.floor(n_orders/100)) == 0:
            stdout.write('\r%.2f%%' % (float(counter*100) / n_orders))
            stdout.flush()

        order = dict(zip(st.order_keys, sing_ord))

        driver_map.incrementAt(order[st.driver_id])
        passenger_map.incrementAt(order[st.passenger_id])

        s_distr_idx = int(dist_map.getkey_or_create(order[st.start_district_hash]))-1
        d_distr_idx = int(dist_map.getkey_or_create(order[st.dest_district_hash]))-1
        week_day = datetime.datetime.strptime(order[st.time], '%Y-%m-%d %H:%M:%S').weekday()
        tslot_idx = int(get_timeslot(order[st.time]))

        if order[st.driver_id] == 'NULL':
            gap[week_day][s_distr_idx][tslot_idx] += 1
        else:
            supply[week_day][s_distr_idx][tslot_idx] += 1
        demand[week_day][s_distr_idx][tslot_idx] += 1
        start_dist[week_day][s_distr_idx] += 1
        if d_distr_idx < st.n_districts:
            dest_dist[week_day][d_distr_idx] += 1

    print '\r'
    # orders = whiten(orders)
    save(st.eval_dir+'gap', gap)
    save(st.eval_dir+'demand', norm(demand))
    save(st.eval_dir+'supply', norm(supply))

    save(st.eval_dir+'start_dist', start_dist)
    save(st.eval_dir+'dest_dist', dest_dist)

def preprocess_weather(date):
    print 'preprocessing weather'
    weather_mat = np.zeros(shape=(st.n_train_days, int(st.n_timeslots), 3))
    for w_file in glob.glob(st.data_dir + 'weather_data_' + date):
        day_idx = int(w_file[-2:])#.replace('-', ''))
        w_data = load_csv(w_file)

        # if st.data_dir == 'data/':
        #     week_day = datetime.datetime.strptime(w_file[-10:], '%Y-%m-%d').weekday()
        # else:
        #     week_day = datetime.datetime.strptime(w_file[-10-st.cut_off_test:-st.cut_off_test], '%Y-%m-%d').weekday()

        for sing_w in w_data:
            w_dict = dict(zip(st.weather_keys, sing_w))
            tslot_idx = int(get_timeslot(w_dict[st.time]))
            w_temp = w_dict[st.w_temperature]
            w_pm25 = w_dict[st.w_pm25]
            w_weather = w_dict[st.w_weather]

            weather_mat[day_idx-1][tslot_idx] = [w_weather, w_temp, w_pm25]

    save(st.eval_dir+'weather', weather_mat)
    smooth_weather()
    smooth_weather_test()

def recursive(ndarray, keys, value, lvl):
    if len(keys) > 1:
        ndarray[keys[0]] = recursive(ndarray[keys[0]], keys[1:], value, lvl+1)
        return ndarray
    ndarray[keys] = value
    return ndarray

def preprocess_pois():
    print 'preprocessing pois'
    dist_map = create_distr_Map()
    poi_map = np.zeros(shape=(st.n_districts, st.n_poi_first, st.n_poi_second))
    # poi_map = np.zeros(shape=(int(st.n_districts), int(st.n_poi_first)))
    pois = merge_files(glob.glob(st.data_dir+'poi_data'))
    for entry in pois:
        distr_hash = entry[0]
        distr_idx = int(dist_map.getkey_or_create(distr_hash))-1

        for p in entry[1:]:
            try:
                p.index(':')
                classes, num = p.split(':')
            except:
                num = 1
                classes = p

            classes = [int(x) for x in classes.split('#')]
            try:
                keys = [distr_idx]+classes
                recursive(poi_map, keys, int(num), 0)
            except:
                raise Exception('More than 2 class levels found: %s %s' % len(classes, p))

    save(st.eval_dir+'pois', norm(poi_map))

def visualizations(interpolate_missing=False):
    # visualize_orders(load(st.eval_dir+'demand.bin'), 'Demand', normalize=True)
    # visualize_orders(load(st.eval_dir+'supply.bin'), 'Supply', normalize=True)
    # visualize_orders(load(st.eval_dir+'gap.bin'), 'Gap', normalize=True)
    # hist(load(st.eval_dir+'start_dist.bin'), 'Start_dist', y_range=[70, 180000])
    # hist(load(st.eval_dir+'dest_dist.bin'), 'Dest_dist', y_range=[70, 120000])

    # if interpolate_missing:
    #     traffic_data = interpolate_traffic(53)
    #     visualize_traffic(traffic_data, 'Traffic', normalize=True)
    # else:
    #     visualize_traffic(load(st.eval_dir + 'traffic.bin'), 'Traffic', normalize=True)

    # visualize_weather(load(st.eval_dir + 'weather.bin'), 'Weather', '(Weather, Temp, PM25)')
    visualize_pois(load(st.eval_dir + 'pois.bin'), 'Pois level 1')

def preprocessing(date='*', interpolate_missing=False):
    # logging.info('Running preprocessing for: %s' % date)
    preprocess_pois()
    # preprocess_weather(date)
    # preprocess_traffic(date, interpolate_missing)
    # preprocess_orders(date)

def prediction_postprocessing(data, gap, prediction_times, n_pred_tisl):
    save_timestmp = toUTCtimestamp(datetime.datetime.utcnow())
    pred_formatted = np.asarray([float('%.2f' % x) for x in data.tolist()]).reshape((st.n_districts, n_pred_tisl))
    visualize_prediction((pred_formatted), 'prediction', n_pred_tisl, save_timestmp)
    pred_shaped = data.reshape((st.n_districts, n_pred_tisl))
    print 'predictions: %s' % pred_formatted
    print 'gap:         %s' % gap.flatten()

    timeslots = [x.split('-')[3] for x in prediction_times]
    gap_subset = np.ndarray((7, st.n_districts, n_pred_tisl))
    for tsl_idx, tslot in enumerate([int(t) for t in timeslots]):
        week_day = datetime.datetime.strptime(prediction_times[tsl_idx][:10], '%Y-%m-%d').weekday()
        gap_subset[week_day, :,tsl_idx] = gap[week_day,:,tslot]

    gap_subset = np.mean(gap_subset, axis=0)
    gap_inv = np.linalg.inv(np.dot(gap_subset.transpose(), gap_subset))
    MAPE = np.sum(np.sum( np.abs(np.dot((gap_subset-pred_shaped), gap_inv)) )/n_pred_tisl )/st.n_districts

    print 'MAPE: %s' % MAPE
    logging.info('MAPE: %s' % MAPE)

    logging.info('saved prediction to file: %s_%s.csv' % (st.eval_dir_test, save_timestmp))
    logging.info('saved prediction to file: %sprediction_%s.png' % (st.eval_dir_test, save_timestmp))
    save_predictions(prediction_times, pred_formatted, save_timestmp)

def training_sgd():
    print 'train model'
    # if interpolate_missing:traffic = interpolate_traffic(53)
    # else:                  traffic = load(st.eval_dir+'traffic.bin')
    traffic = load(st.eval_dir+'traffic.bin')
    demand_test = load(st.eval_dir_test+'demand.bin')
    demand_train = load(st.eval_dir+'demand.bin')
    supply_test = load(st.eval_dir_test+'supply.bin')
    supply_train = load(st.eval_dir+'supply.bin')
    gap = load(st.eval_dir+'gap.bin')
    pois = load(st.eval_dir+'pois.bin') #[:,:-15]

    linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    regr = linear_model.LinearRegression()

    sample_d_t = []
    gap_d_t= []
    for d in range(st.n_districts):
        for week_day in range(7):
            for dtime_slt in range(st.n_timeslots):
                dem = demand_test[week_day,d,dtime_slt]
                if math.isnan(dem): dem = demand_train[week_day, d, dtime_slt]
                supp = supply_test[week_day,d,dtime_slt]
                if math.isnan(supp):  supp = supply_train[week_day,d,dtime_slt]
                params = [week_day, dtime_slt, dem, supp]
                sample_d_t.append(np.concatenate((params, traffic[week_day, d, dtime_slt, :].flatten(), pois.flatten()), axis=0))
                gap_d_t.append(gap[week_day, d, dtime_slt])

    print 'train: %s ; %s' % (np.asarray(sample_d_t).shape, np.asarray(gap_d_t).shape)
    regr.fit(sample_d_t, gap_d_t)
    # print 'coeff %s' % model.coef_
    return regr

def build_training_data():
    # traffic = load(st.eval_dir_test+'traffic.bin')
    # if interpolate_missing:
    #     traffic = interpolate_traffic(53)
    # else:
    #     traffic = load(st.eval_dir_test+'traffic.bin')
    # traffic_train = np.asarray([norm(jam) for jam in load(st.eval_dir+'traffic.bin')])
    traffic_train = load(st.eval_dir+'traffic.bin')
    traffic_test = load(st.eval_dir_test+'traffic.bin')

    destination = load(st.eval_dir+'dest_dist.bin')
    start = load(st.eval_dir + 'start_dist.bin')
    weather_train = load(st.eval_dir+'weather.bin')
    weather_test = load(st.eval_dir_test+'weather.bin')

    demand_test = load(st.eval_dir_test+'demand.bin')
    demand_train = load(st.eval_dir+'demand.bin')
    supply_test = load(st.eval_dir_test+'supply.bin')
    supply_train = load(st.eval_dir+'supply.bin')
    gap = load(st.eval_dir+'gap.bin')
    gap_test = load(st.eval_dir_test+'gap.bin')
    pois = load(st.eval_dir+'pois.bin')[:,:-15]

    prediction_times = []
    with open(st.data_dir_test + 'read_me_1.txt') as f:
        prediction_date = f.read().splitlines()
    for p in prediction_date:
        prediction_times.append(p)
    prediction_times = prediction_times[st.n_csv_header_lines:]
    timeslots = [x.split('-')[3] for x in prediction_times]
    n_pred_tisl = len(timeslots)

    sample_d_t = []
    gap_d_t = []
    for d in range(st.n_districts):
        for week_day in range(7):
            for dtime_slt in range(st.n_timeslots):
                dem = demand_test[week_day, d, dtime_slt]
                if math.isnan(dem): dem = demand_train[week_day, d, dtime_slt]
                supp = supply_test[week_day, d, dtime_slt]
                if math.isnan(supp):  supp = supply_train[week_day, d, dtime_slt]
                params = [week_day, dtime_slt, dem, supp]
                sample_d_t.append(np.concatenate((params,
                                                  traffic_train[week_day, d, dtime_slt, :].flatten(),
                                                  pois.flatten(),
                                                  destination[week_day, d].flatten(),
                                                  start[week_day, d].flatten(),
                                                  weather_train[week_day, dtime_slt,:].flatten()
                                                  ), axis=0))
                gap_d_t.append(gap[week_day, d, dtime_slt])

    sample_d_t_test = []
    gap_d_t_test = []
    for d in range(st.n_districts):
        for dtime_slt in range(n_pred_tisl):
            week_day = datetime.datetime.strptime(prediction_times[dtime_slt][:10], '%Y-%m-%d').weekday()
            dem = demand_test[week_day, d, dtime_slt]
            if math.isnan(dem): dem = demand_train[week_day, d, dtime_slt]
            supp = supply_test[week_day, d, dtime_slt]
            if math.isnan(supp):  supp = supply_train[week_day, d, dtime_slt]
            params = [week_day, dtime_slt, dem, supp]
            sample_d_t_test.append(np.concatenate((params,
                                                   traffic_test[week_day, d, dtime_slt, :].flatten(),
                                                   pois.flatten(),
                                                   destination[week_day, d].flatten(),
                                                   start[week_day, d].flatten(),
                                                   weather_test[week_day, dtime_slt,:].flatten()
                                                   ), axis=0))
            gap_d_t_test.append(gap_test[week_day, d, dtime_slt])
    return gap, sample_d_t, sample_d_t_test, gap_d_t, gap_d_t_test, prediction_times, n_pred_tisl


def prediction_sgd(model, interpolate_missing=False):
    print 'predict values'
    prediction_times = []
    with open(st.data_dir_test+'read_me_1.txt') as f:
        prediction_date = f.read().splitlines()
    for p in prediction_date:
        prediction_times.append(p)
    prediction_times = prediction_times[st.n_csv_header_lines:]
    timeslots = [x.split('-')[3] for x in prediction_times]
    n_pred_tisl = len(timeslots)

    # if interpolate_missing:traffic = interpolate_traffic(53)
    # else:                  traffic = load(st.eval_dir_test+'traffic.bin')
    traffic = load(st.eval_dir + 'traffic.bin')
    demand_test = load(st.eval_dir_test+'demand.bin')
    demand_train = load(st.eval_dir+'demand.bin')
    supply_test = load(st.eval_dir_test+'supply.bin')
    supply_train = load(st.eval_dir+'supply.bin')
    gap = load(st.eval_dir_test+'gap.bin')
    pois = np.array(load(st.eval_dir_test+'pois.bin'), dtype=float) #[:,:-15]

    sample_d_t = []
    for d in range(st.n_districts):
        for dtime_slt in range(n_pred_tisl):
            week_day = datetime.datetime.strptime(prediction_times[dtime_slt][:10], '%Y-%m-%d').weekday()
            dem = demand_test[week_day, d, dtime_slt]
            if math.isnan(dem): dem = demand_train[week_day, d, dtime_slt]
            supp = supply_test[week_day, d, dtime_slt]
            if math.isnan(supp):  supp = supply_train[week_day, d, dtime_slt]
            params = [week_day, dtime_slt, dem, supp]
            sample_d_t.append(np.concatenate((params, traffic[week_day, d, dtime_slt, :].flatten(), pois.flatten()), axis=0))

    s = model.predict(sample_d_t)#  * 0.98  # TODO remove MAPE factor
    print s.shape
    prediction_postprocessing(s, gap, prediction_times, n_pred_tisl)

def train_nn(interpolate_missing=False):
    gap, sample_d_t, sample_d_t_test, gap_d_t, gap_d_t_test, prediction_times, n_pred_tisl= build_training_data()
    cut = 10000
    tr = [np.asarray(sample_d_t[:-cut]), np.asarray(gap_d_t[:-cut])]
    print 'train: %s  %s'  % (tr[0].shape, tr[1].shape)
    va = [np.asarray(sample_d_t[-cut:]), np.asarray(gap_d_t[-cut:])]
    te = [np.asarray(sample_d_t_test), np.asarray(gap_d_t_test)]

    classifier = mlp_train(logging, tr, va, te)

    print '... prediction'
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred,
        on_unused_input='warn'
    )

    prediction = predict_model(np.asarray(sample_d_t_test, dtype=theano.config.floatX))# * 0.98  # TODO remove MAPE factor
    print 'predition # results: %s ' % prediction.shape
    prediction_postprocessing(prediction, gap, prediction_times, n_pred_tisl)

if __name__ == "__main__":
    date = '2016-01-*'

    # started_at = datetime.datetime.now()
    # logging.info('------')
    # logging.info('Started at: %s' % started_at)
    # print(started_at)

    preprocessing(date, interpolate_missing=False)
    visualizations()

    # model = training_sgd()
    # prediction_sgd(model)
    # train_nn()

    # finished_at = datetime.datetime.now()
    # print(finished_at)
    # logging.info('Finished at: %s' % finished_at)