#!/usr/bin/env /home/dominik/anaconda2/bin/python

import numpy as np
import itertools
import statics as st
from utils_file import load_model, save_model
from utils_image import plot_receptive_fields
from utils_date import toUTCtimestamp
from datetime import datetime
import theano
import theano.tensor as T
import climin
import climin.initialize
import climin.util

class LinearRegression(object):

    def __init__(self, input, W=None, b=None):
        self.input = input
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.y_pred = T.sum(T.dot(self.input, self.W) + self.b, axis=1)

    def mean_abs_percentage_error(self, x, y):
        self.input = x
        return T.mean(abs(T.dot(y - self.y_pred, T.inv(y))))

    def error(self, y):
        return T.sum(T.pow(y - self.y_pred, 2) / 2*y.shape[0])

class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out, active_func_name, W_hidden1=None, b_hidden1=None, W_hidden2=None, b_hidden2=None, W_log=None, b_log=None):

        activ_func = T.tanh
        if active_func_name == 'Logistic sigmoid':
            activ_func = T.nnet.sigmoid
        elif active_func_name == 'Rectified linear unit':
            activ_func = T.nnet.relu

        self.hiddenLayer1 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            W = W_hidden1,
            b = b_hidden1,
            activation=activ_func
        )

        self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer1.output,
            n_in=n_hidden,
            n_out=n_hidden,
            W=W_hidden2,
            b=b_hidden2,
            activation=activ_func
        )

        self.outputLayer = LinearRegression(
            # input=self.hiddenLayer1.output,
            input = self.hiddenLayer2.output,
            W=W_log,
            b=b_log
        )

        self.L1 = (
            abs(self.hiddenLayer1.W).sum()
            + abs(self.hiddenLayer2.W).sum()
            + abs(self.outputLayer.W).sum()
        )

        self.L2_sqr = (
              (self.hiddenLayer1.W ** 2).sum()
            + (self.hiddenLayer2.W ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
        )

        self.error = self.outputLayer.error
        self.y_pred = self.outputLayer.y_pred
        self.mean_abs_percentage_error = self.outputLayer.mean_abs_percentage_error
        self.params = self.hiddenLayer1.params + self.hiddenLayer2.params + self.outputLayer.params
        self.input = input

def mlp_train(logging, data_train, data_validate, data_test, add_L1_L2_regularizer=False):
    train_set_x, train_set_y = data_train
    valid_set_x, valid_set_y = data_validate
    test_set_x, test_set_y = data_test

    batch_size = 5000
    n_in = train_set_x.shape[1]
    n_out = 200  #n_in    #batch_size
    n_hidden = 100
    n_epochs = 20
    opt_name = 'RmsProp'    #'Adadelta'
    active_func_name = 'tanh'   # 'Rectified linear unit'
    n_train_batches = train_set_x.shape[0] // batch_size
    print 'training %i epochs' % n_epochs

    rng = np.random.RandomState(1234)
    tmpl = [(n_in, n_hidden), n_hidden, (n_hidden, n_hidden), n_hidden,(n_hidden, n_out), n_out]
    wrt_flat, (Weights_hidden1, bias_hidden1, Weights_hidden2, bias_hidden2, Weights_log, bias_log) = climin.util.empty_with_views(tmpl)
    # tmpl = [(n_in, n_hidden), n_hidden, (n_hidden, n_out), n_out]
    # wrt_flat, (Weights_hidden1, bias_hidden1, Weights_log, bias_log) = climin.util.empty_with_views(tmpl)
    variance = 1
    climin.initialize.randomize_normal(wrt_flat, 0, variance)
    logging.info('Weight initialization variance: %s' % (variance))

    print 'tmpl: %s ' % tmpl
    logging.info('Batch size: %s' % batch_size)
    logging.info('Hidden layer: %s' % n_hidden)
    logging.info('Epochs: %s' % n_epochs)
    logging.info('Reg. optimizer: %s' % opt_name)
    logging.info('Activation function: %s' % active_func_name)
    logging.info('Tmpl: %s' % tmpl)

    x = T.matrix('x')
    y = T.ivector('y')

    classifier = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out,
        active_func_name=active_func_name,
        W_hidden1 = theano.shared(value=Weights_hidden1, name='Wh', borrow=True),
        b_hidden1 = theano.shared(value=bias_hidden1, name='bh', borrow=True),
        W_hidden2=theano.shared(value=Weights_hidden2, name='Wh', borrow=True),
        b_hidden2=theano.shared(value=bias_hidden2, name='bh', borrow=True),
        W_log = theano.shared(value=Weights_log, name='Wo', borrow=True),
        b_log = theano.shared(value=bias_log, name='bo', borrow=True)
    )

    if add_L1_L2_regularizer:
        L1_reg = 0.001
        L2_reg = 0.0001
        cost_reg = (classifier.error(y)
                   + L1_reg * classifier.L1
                   + L2_reg * classifier.L2_sqr
        )
        logging.info('Loss L1/L2 regularizer L1: %s L2: %s' % (L1_reg, L2_reg))
    else:
        cost_reg = classifier.error(y)
        logging.info('Loss L1/L2 regularizer: None')

    gradients = theano.function(
        inputs=[x, y],
        outputs=T.grad(cost_reg, classifier.params),
        allow_input_downcast=True
    )

    try:
        save_time_delta = toUTCtimestamp(datetime.utcnow())
        latest, classifier = load_model(classifier)
        print '... load latest model: %s' % latest
        logging.info('... load latest model: %s' % latest)
    except ImportWarning as ex:
        print ex.message
        logging.info(ex.message)


    # latest, classifier = load_model(classifier)
    # plot_receptive_fields(classifier, latest)
    # return


    def d_loss_wrt_pars(parameters, inputs, targets):
        g_W_h, g_b_h, g_W_h2, g_b_h2, g_W_l, g_b_l = gradients(inputs, targets)
        return np.concatenate([g_W_h.flatten(), g_b_h, g_W_h2.flatten(), g_b_h2, g_W_l.flatten(), g_b_l])
        g_W_h, g_b_h, g_W_l, g_b_l = gradients(inputs, targets)
        return np.concatenate([g_W_h.flatten(), g_b_h, g_W_l.flatten(), g_b_l])

    mean_abs_percentage_error = theano.function(
        inputs=[x,y],
        outputs=classifier.mean_abs_percentage_error(x, y),
        allow_input_downcast=True
    )

    zero_one_loss = theano.function(
        inputs = [x, y],
        outputs = classifier.error(y),
        allow_input_downcast = True
    )

    print('... training: hidden layer act func: %s, lin reg optimizer: %s' % (active_func_name, opt_name))

    train_loss = []
    valid_loss = []
    test_loss = []

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.

    if batch_size is None:
        args = itertools.repeat(([train_set_x, train_set_y], {}))
    else:
        args = climin.util.iter_minibatches([train_set_x, train_set_y], batch_size, [0, 0])
        args = ((i, {}) for i in args)

    print('... using linear regression optimizer: %s' % opt_name)
    if opt_name == 'RmsProp':
        step = 1e-4
        dec = 0.9
        opt = climin.RmsProp(wrt_flat, d_loss_wrt_pars, step_rate=step, decay=dec, args=args)
        logging.info('RmsProp step rate: %s, decay %s' % (step, dec))
    elif opt_name == 'Rprop':
        opt = climin.Rprop(wrt_flat, d_loss_wrt_pars, args=args)
    elif opt_name == 'Adam':
        opt = climin.Adam(wrt_flat, d_loss_wrt_pars, step_rate=0.0002, decay=0.99999999, decay_mom1=0.1,
                          decay_mom2=0.001, momentum=0, offset=1e-08, args=args)
    elif opt_name == 'Adadelta':
        step=1
        dec=0.9
        mom=.95
        offs=0.0001
        opt = climin.Adadelta(wrt_flat, d_loss_wrt_pars, step_rate=step, decay=dec, momentum=mom, offset=offs, args=args)
        logging.info('Adadelta step rate: %s, decay: %s, momentum: %s, offset: %s' % (step, dec, mom, offs))

    started_at = datetime.now()
    logging.info('Started at: %s' % started_at)
    print('Started at: %s' % started_at)
    step_rate = None

    for info in opt:
        iter = info['n_iter']
        epoch = iter // n_train_batches

        if toUTCtimestamp(datetime.utcnow()) - save_time_delta > st.save_model_time_delta and epoch % st.save_model_freq == 0:
            save_model(logging, classifier)
            save_time_delta = toUTCtimestamp(datetime.utcnow())

        if (iter + 1) % validation_frequency == 0:
            valid_loss.append(zero_one_loss(valid_set_x, valid_set_y))
            train_loss.append(zero_one_loss(train_set_x, train_set_y))
            train_score = np.mean(train_loss)
            this_validation_loss = np.mean(valid_loss)

            print('epoch %i, validation error %f %%, train error %f %%' % (epoch, this_validation_loss * 100., train_score*100.))
            logging.info('epoch %i, validation error %f %%' % (epoch, this_validation_loss * 100.))

            if this_validation_loss < best_validation_loss:
                if this_validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter

                test_loss.append(zero_one_loss(test_set_x, test_set_y))
                test_score = np.mean(test_loss)

                print(
                    (
                        '     epoch %i, test error of'
                        ' best model %f %%'
                    ) %
                    (
                        epoch,
                        test_score * 100.
                    )
                )
                logging.info(('     epoch %i, test error of'' best model %f %%') % (epoch,test_score * 100.))

        if epoch >= n_epochs:   # patience <= iter or
            step_rate = info.get('step_rate')
            break

    print 'finished at step_rate: %f' % step_rate
    finished_at = datetime.now()
    print('Finished at: %s' % finished_at)
    logging.info('Finished at: %s' % finished_at)

    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    logging.info(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    mape = mean_abs_percentage_error(test_set_x, test_set_y)
    print('MAPE: %s' % mape)
    logging.info('MAPE: %s' % mape)

    return classifier
