#!/usr/bin/env /home/dominik/anaconda2/bin/python

import math
import theano
from theano import tensor as T
from theano import function as tfunc
from sklearn import linear_model
from utils_file import *
from taxi import prediction_postprocessing

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            W_values = np.zeros((n_in, n_out), dtype = theano.config.floatX)
            W = theano.shared(value = W_values, name = 'W', borrow = True )

        if b is None:
            b_values = np.zeros((n_out,), dtype = theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True )

        self.W = W
        self.b = b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

empty_train_days = [3, 7, 11, 15, 19, 24]
empty_test_days = [23, 25, 27, 29]

def normalize(data):
    eps_norm = 1  # 10 proposed for value range of 255
    print('... normalize input eps_norm: %s ' % eps_norm)
    # logger.info('... normalizing input (eps_norm: %s) ' % eps_norm)
    data = np.asarray(data)
    means = data.mean(axis=1, keepdims=True)
    x_mean = data - means
    variance = np.var(data, axis=1, keepdims=False)
    return x_mean / (np.sqrt(variance + eps_norm))[:, np.newaxis]


def whiten(data):
    eps = 0.1
    print('... whiten input (eps: %s)' % eps)
    # logger.info('... whiten input (eps: %s)' % eps)
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

def training_sgd(logging):
    logging.info('Sklearn - Linear Regression')
    logging.info('... train model')
    print 'train model'
    # if interpolate_missing:traffic = interpolate_traffic(53)
    # else:                  traffic = load(st.eval_dir+'traffic.bin')
    # traffic = load(st.eval_dir + 'traffic.bin')
    # demand_test = load(st.eval_dir_test + 'demand.bin')
    # demand_train = load(st.eval_dir + 'demand.bin')
    # supply_test = load(st.eval_dir_test + 'supply.bin')
    # supply_train = load(st.eval_dir + 'supply.bin')
    # gap = load(st.eval_dir + 'gap.bin')
    # pois = load(st.eval_dir + 'pois.bin')  # [:,:-15]

    demand_train = load(st.eval_dir + 'demand_daywise.bin')
    supply_train = load(st.eval_dir + 'supply_daywise.bin')
    gap_train = load(st.eval_dir + 'gap_daywise.bin')
    start_train = load(st.eval_dir + 'start_dist_daywise.bin')
    dest_train = load(st.eval_dir + 'dest_dist_daywise.bin')
    traffic_train = load(st.eval_dir + 'traffic_daywise.bin')
    weather_train = load(st.eval_dir + 'weather_daywise.bin')
    pois = load(st.eval_dir_test + 'pois.bin')
    # pois = load(st.eval_dir_test + 'pois_simple.bin')

    linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    regr = linear_model.LinearRegression()

    sample_d_t = []
    gap_d_t = []
    # for d in range(st.n_districts):
    #     for week_day in range(7):
    #         for dtime_slt in range(st.n_timeslots):
    #             dem = demand_test[week_day, d, dtime_slt]
    #             if math.isnan(dem): dem = demand_train[week_day, d, dtime_slt]
    #             supp = supply_test[week_day, d, dtime_slt]
    #             if math.isnan(supp):  supp = supply_train[week_day, d, dtime_slt]
    #             params = [week_day, dtime_slt, dem, supp]
    #             sample_d_t.append(
    #                 np.concatenate((params, traffic[week_day, d, dtime_slt, :].flatten(), pois.flatten()), axis=0))
    #             gap_d_t.append(gap[week_day, d, dtime_slt])

    for distr in range(st.n_districts):
        for dtime_slt in range(st.n_timeslots):
            for day in range(len(gap_train)):
                skip_day = True
                try:
                    empty_train_days.index(day)
                except:
                    skip_day = False
                if skip_day or day > 30:
                    continue

                sample_d_t.append(np.concatenate(([day, dtime_slt],
                                                     traffic_train[day, distr, dtime_slt, :].flatten(),
                                                     pois[distr].flatten(),
                                                     dest_train[day, distr].flatten(),
                                                     start_train[day, distr].flatten(),
                                                     demand_train[day, distr, dtime_slt].flatten(),
                                                     supply_train[day, distr, dtime_slt].flatten(),
                                                     weather_train[day, :, dtime_slt].flatten()
                                                     ), axis=0))
                gap_d_t.append(gap_train[day, distr, dtime_slt])

    sample_d_t = normalize(sample_d_t).transpose()
    sample_d_t = whiten(sample_d_t).transpose()

    print 'train stats: %s %s' % (np.mean(sample_d_t), np.var(sample_d_t))

    print 'train: %s ; %s' % (np.asarray(sample_d_t).shape, np.asarray(gap_d_t).shape)
    regr.fit(sample_d_t, gap_d_t)
    # print 'coeff %s' % model.coef_
    return regr


def prediction_sgd(model, logging, interpolate_missing=False):
    print 'predict values'
    logging.info('... predict values')
    prediction_times = []
    with open(st.data_dir_test+'read_me_all.txt') as f:
        prediction_date = f.read().splitlines()
    for p in prediction_date:
        prediction_times.append(p)
    prediction_times = prediction_times[st.n_csv_header_lines:]
    timeslots = [x.split('-')[3] for x in prediction_times]
    n_pred_tisl = len(timeslots)

    # if interpolate_missing:traffic = interpolate_traffic(53)
    # else:                  traffic = load(st.eval_dir_test+'traffic.bin')
    # traffic = load(st.eval_dir + 'traffic.bin')
    # demand_test = load(st.eval_dir_test+'demand.bin')
    # demand_train = load(st.eval_dir+'demand.bin')
    # supply_test = load(st.eval_dir_test+'supply.bin')
    # supply_train = load(st.eval_dir+'supply.bin')
    # gap = load(st.eval_dir_test+'gap.bin')
    # pois = np.array(load(st.eval_dir_test+'pois.bin'), dtype=float) #[:,:-15]
    #
    # sample_d_t = []
    # for d in range(st.n_districts):
    #     for dtime_slt in range(n_pred_tisl):
    #         week_day = datetime.datetime.strptime(prediction_times[dtime_slt][:10], '%Y-%m-%d').weekday()
    #         dem = demand_test[week_day, d, dtime_slt]
    #         if math.isnan(dem): dem = demand_train[week_day, d, dtime_slt]
    #         supp = supply_test[week_day, d, dtime_slt]
    #         if math.isnan(supp):  supp = supply_train[week_day, d, dtime_slt]
    #         params = [week_day, dtime_slt, dem, supp]
    #         sample_d_t.append(np.concatenate((params, traffic[week_day, d, dtime_slt, :].flatten(), pois.flatten()), axis=0))


    demand_test = load(st.eval_dir_test + 'demand_daywise.bin')
    supply_test = load(st.eval_dir_test + 'supply_daywise.bin')
    gap_test = load(st.eval_dir_test + 'gap_daywise.bin')
    start_test = load(st.eval_dir_test + 'start_dist_daywise.bin')
    dest_test = load(st.eval_dir_test + 'dest_dist_daywise.bin')
    traffic_test = load(st.eval_dir_test + 'traffic_daywise.bin')
    weather_test = load(st.eval_dir_test + 'weather_daywise_test.bin')
    pois = load(st.eval_dir_test + 'pois.bin')
    # pois = load(st.eval_dir_test + 'pois_simple.bin')

    prediction_times = []
    with open(st.data_dir_test + 'read_me_all.txt') as f:
        prediction_date = f.read().splitlines()
    for p in prediction_date:
        prediction_times.append(p)
    prediction_times = prediction_times[st.n_csv_header_lines:]
    pred_days = [x.split('-')[2] for x in prediction_times]
    pred_timeslots = [dict(zip(st.prediction_keys, x.split('-'))) for x in prediction_times]
    n_pred_tisl = len(pred_timeslots)

    samples_test = []
    for distr in range(st.n_districts):
        for pred_idx, pred_dict in enumerate(pred_timeslots):
            day = int(pred_dict.get('day'))
            dtime_slt = int(pred_dict.get('timeslot'))
            skip_day = True
            try:
                empty_test_days.index(day)
            except:
                skip_day = False
            if skip_day or day > 30:
                continue

            samples_test.append(np.concatenate(([day, dtime_slt],
                                                traffic_test[day, distr, dtime_slt, :].flatten(),
                                                pois[distr].flatten(),
                                                dest_test[day, distr].flatten(),
                                                start_test[day, distr].flatten(),
                                                demand_test[day, distr, dtime_slt].flatten(),
                                                supply_test[day, distr, dtime_slt].flatten(),
                                                weather_test[day, :, dtime_slt].flatten()
                                                ), axis=0))

    samples_test = normalize(samples_test).transpose()
    samples_test = whiten(samples_test).transpose()

    print 'prediction stats: %s %s' % (np.mean(samples_test), np.var(samples_test))

    s = model.predict(samples_test)
    print s.shape
    prediction_postprocessing(s, gap_test, prediction_times, n_pred_tisl)
