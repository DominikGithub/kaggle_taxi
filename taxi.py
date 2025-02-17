#!/usr/bin/env /home/dominik/anaconda2/bin/python

from sys import stdout
import logging
import numpy as np
from random import randint
from LogisticRegression import *
from learning_data_builder import Learning_data_builder
from mlp import mlp_train
from test_RNN import test_RNN
from data_container import IncMap, create_distr_Map
from utils_file import *
from utils_image import *
from utils_date import *
import pandas as pd
from correlation_smoothing import smooth_visualize_weather_train, smooth_visualize_weather_test, interpolate_traffic
from preprocessor import Preprocessor

logging.basicConfig(filename='taxi.log', level=logging.INFO)

def preprocess_traffic(date, interpolate_missing=False):
    print 'preprocessing traffic'
    dist_map = create_distr_Map()
    traff_map = np.zeros(shape=(7, int(st.n_districts), int(st.n_timeslots), int(st.max_congestion_lvls)))

    for traffic_file in glob.glob(st.data_dir+'traffic_data_'+date):
        traffic_data = load_csv(traffic_file)

        if st.data_dir == 'data_train/':
            week_day = datetime.strptime(traffic_file[-10:], '%Y-%m-%d').weekday()
        else:
            week_day = datetime.strptime(traffic_file[-10-st.cut_off_test:-st.cut_off_test], '%Y-%m-%d').weekday()

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
        week_day = datetime.strptime(order[st.time], '%Y-%m-%d %H:%M:%S').weekday()
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
    save(st.eval_dir+'demand', demand)
    save(st.eval_dir+'supply', supply)

    save(st.eval_dir+'start_dist', start_dist)
    save(st.eval_dir+'dest_dist', dest_dist)

def preprocess_weather(date):
    print 'preprocessing weather'
    weather_mat = np.zeros(shape=(st.n_train_days, int(st.n_timeslots), 3))
    for w_file in glob.glob(st.data_dir + 'weather_data_' + date):
        day_idx = int(w_file[-2:])#.replace('-', ''))
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

            weather_mat[day_idx-1][tslot_idx] = [w_weather, w_temp, w_pm25]

    save(st.eval_dir+'weather', weather_mat)
    smooth_visualize_weather_train()
    smooth_visualize_weather_test()

def poi_map_recursive(ndarray, keys, value):
    if len(keys) > 1:
        ndarray[keys[0]] = poi_map_recursive(ndarray[keys[0]], keys[1:], value)
        return ndarray
    if len(keys) == 1:
        ndarray[keys[0]] = value
    return ndarray

def preprocess_pois():
    print 'preprocessing pois'
    dist_map = create_distr_Map()
    poi_map = np.zeros(shape=(st.n_districts, st.n_poi_first, st.n_poi_second, st.n_poi_third))
    poi_map_simple = np.zeros(shape=(int(st.n_districts), int(st.n_poi_first)))
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
            poi_map_simple[distr_idx, classes[0]] += int(num)
            try:
                keys = [distr_idx]+classes
                poi_map_recursive(poi_map, keys, int(num))
            except:
                raise Exception('class level outside poi_map size: %s' % p) #9#1#1:14

    save(st.eval_dir+'pois_simple', poi_map_simple)
    save(st.eval_dir+'pois', poi_map)

def preprocessing(date='*', interpolate_missing=False):
    logging.info('Running preprocessing for: %s' % date)
    preprocess_pois()
    preprocess_weather(date)
    preprocess_traffic(date, interpolate_missing)
    preprocess_orders(date)

def prediction_postprocessing(data, gap, prediction_times, n_pred_tisl, save_timestmp):
    data += abs(np.min(data))
    pred_formatted = np.asarray([float('%.2f' % x) for x in data.tolist()]).reshape((st.n_districts, n_pred_tisl))
    save_predictions(prediction_times, pred_formatted, save_timestmp)
    visualize_prediction((pred_formatted), 'prediction', n_pred_tisl, save_timestmp)
    pred_shaped = data.reshape((st.n_districts, n_pred_tisl))
    print 'predictions: %s' % pred_formatted
    print 'gap:         %s' % gap.flatten()

    timeslots = [x.split('-')[3] for x in prediction_times]
    gap_subset = np.ndarray((7, st.n_districts, n_pred_tisl))
    for tsl_idx, tslot in enumerate([int(t) for t in timeslots]):
        week_day = datetime.strptime(prediction_times[tsl_idx][:10], '%Y-%m-%d').weekday()
        gap_subset[week_day, :,tsl_idx] = gap[week_day,:,tslot]

    gap_subset = np.mean(gap_subset, axis=0)
    gap_inv = np.linalg.inv(np.dot(gap_subset.transpose(), gap_subset))
    MAPE = np.sum(np.sum( np.abs(np.dot((gap_subset-pred_shaped), gap_inv)) )/n_pred_tisl )/st.n_districts
    print 'MAPE: %s' % MAPE
    logging.warn('MAPE: %s' % MAPE)

    logging.info('saved prediction to file: %s_%s.csv' % (st.eval_dir_test, save_timestmp))
    logging.info('saved prediction to file: %sprediction_%s.png' % (st.eval_dir_test, save_timestmp))
    os.system('espeak "your program has finished"')

def shuffel_data(samples, labels):
    print '... shuffeling data'
    logging.info('... shuffeling data')
    rand_idx = randint(0, len(samples))
    pattern_first_sample = samples[rand_idx]
    pattern_first_gap = labels[rand_idx]
    data_pair = np.column_stack([samples, np.asarray(labels)])
    np.random.shuffle(data_pair)
    samples = data_pair[:,:-1]
    labels = data_pair[:,-1:].flatten()

    found_match = False
    for idx, sample in enumerate(samples):
        if np.array_equal(pattern_first_sample, sample):
            # print str(idx)+': '+str(pattern_first_gap)+'  '+str(labels[idx])
            if pattern_first_gap == labels[idx]:
                found_match = True
            break
    if not found_match:
        logging.info('... shuffeling data failed to validate')
        raise Exception('shuffeling data failed to validate')
    return samples, labels

def train_nn(interpolate_missing=False):
    builder = Learning_data_builder(logging)
    sample_train, sample_test, gap_train, gap_test, prediction_times, n_pred_tisl = builder.build_daily_training_data()

    valid_size = int(np.floor(len(sample_train) *0.3))
    print 'valid_size: %s' % valid_size

    classifier = None
    for shuffel in range(st.n_times_shuffel):
        print 'shuffeling: %i of %i' % (shuffel, st.n_times_shuffel)
        logging.info('shuffeling: %i of %i' % (shuffel, st.n_times_shuffel))

        sample_train, gap_train = shuffel_data(sample_train, gap_train)
        tr = [np.asarray(sample_train[valid_size:]), np.asarray(gap_train[valid_size:])]
        va = [np.asarray(sample_train[:valid_size]), np.asarray(gap_train[:valid_size])]
        # tr = [np.asarray(sample_train[:-valid_size]), np.asarray(gap_train[:-valid_size])]
        # va = [np.asarray(sample_train[-valid_size:]), np.asarray(gap_train[-valid_size:])]
        te = [np.asarray(sample_test), np.asarray(gap_test)]
        print 'train: %s  %s' % (tr[0].shape, tr[1].shape)


        classifier = mlp_train(logging, tr, va, te, add_L1_L2_regularizer=True)
        # train_set_x, train_set_y = tr
        # classifier = test_RNN(train_set_x, train_set_y, nh=300, nl=3, n_in=train_set_x.shape[1])

        save_model(logging, classifier)

    assert classifier is not None, 'Classifier was not initialized while training'
    return classifier

def MAPE(actuals, predictions):
    """ Returns the Mean Absolute Percentage Error (MAPE) for the predictions of the Supply-Demand gap.
    """
    if len(actuals) != len(predictions):
        return "Error! Actuals and predictions are of different lengths."
    result = 0.0
    df = pd.DataFrame()
    df['actuals'] = actuals
    df['predictions'] = predictions
    df = df[df['actuals'] != 0]
    df['error'] = abs((df['actuals'] - df['predictions']) / df['actuals'])
    result = df['error'].sum() / df['error'].size
    return result

def predict_nn(classifier):
    print '... prediction'
    mape_factor_active = False

    builder = Learning_data_builder(logging)
    sample_test, gap_test, prediction_times, n_pred_tisl = builder.get_daily_test_data()

    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred,
        on_unused_input='warn'
    )

    prediction = predict_model(np.asarray(sample_test, dtype=theano.config.floatX))
    if mape_factor_active:
        mape_factor = 0.98
        prediction *= mape_factor
        logging.info('mape factor: %i' % mape_factor)

    # print 'pMape: %s' % MAPE(prediction, gap_test)
    # logging.info('pMape: %s' % MAPE(prediction, gap_test))

    save_timestmp = toUTCtimestamp(datetime.utcnow())
    dot_idx = str(save_timestmp).index('.')
    save_timestmp = str(save_timestmp)[:dot_idx]
    prediction_postprocessing(prediction, np.asarray(gap_test), prediction_times, n_pred_tisl, save_timestmp)
    # diff_prediction_gap(gap_test, prediction, n_pred_tisl, save_timestmp)
    plot_receptive_fields(classifier, save_timestmp)

def diff_prediction_gap(gap, prediction, timeslots, timestmp):
    print '... plotting gap vs prediction'
    diff = np.asarray([j - i for i, j in zip(gap, prediction)]).reshape((66, 43))
    visualize_prediction(diff, 'diff gap vs prediction', timeslots, timestmp)

if __name__ == "__main__":
    date = '2016-01-*'

    # Preprocessor()
    logging.info('------')
    classifier = train_nn()
    predict_nn(classifier)
    # model = training_sgd(logging)
    # prediction_sgd(model, logging)
