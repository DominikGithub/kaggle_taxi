#!/usr/bin/env /home/dominik/anaconda2/bin/python

import numpy as np
import itertools
from sys import stdout
import theano
import theano.tensor as T
import climin
import climin.initialize
import climin.util

class LinearRegression(object):

    def __init__(self, input, W=None, b=None):
        self.W = W
        self.b = b
        self.p_y_given_x = T.dot(input, self.W) + self.b

        self.y_pred = self.p_y_given_x[:,0]

        self.params = [self.W, self.b]

    # def negative_log_likelihood(self, y):
    #     return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        return T.sum(T.sqr(y-self.y_pred))


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
    def __init__(self, rng, input, n_in, n_hidden, n_out, active_func_name,
                        W_hidden=None, b_hidden=None, W_log=None, b_log=None):

        activ_func = T.tanh
        if active_func_name == 'Logistic sigmoid':
            activ_func = T.nnet.sigmoid
        elif active_func_name == 'Rectified linear unit':
            activ_func = T.nnet.relu

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            W = W_hidden,
            b = b_hidden,
            activation=activ_func
        )

        self.outputLayer = LinearRegression(
            input = self.hiddenLayer.output,
            W=W_log,
            b=b_log
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.outputLayer.W).sum()
        )

        self.L2_sqr = (
              (self.hiddenLayer.W ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
        )

        # self.negative_log_likelihood = self.outputLayer.negative_log_likelihood
        self.errors = self.outputLayer.errors
        self.y_pred = self.outputLayer.y_pred
        self.params = self.hiddenLayer.params + self.outputLayer.params
        self.input = input

def mlp_train(logging, data_train, data_validate, data_test):
    train_set_x, train_set_y = data_train
    valid_set_x, valid_set_y = data_validate
    test_set_x, test_set_y = data_test

    batch_size = 14000
    n_in = train_set_x.shape[1]
    n_out = batch_size
    n_hidden = 800
    n_epochs = 10
    opt_name = 'RmsProp'    #'GradientDescent'
    active_func_name = 'Rectified linear unit'  #'T.tanh'
    n_train_batches = train_set_x.shape[0] // batch_size

    rng = np.random.RandomState(1234)
    tmpl = [(n_in, n_hidden), n_hidden, (n_hidden, n_out), n_out]
    wrt_flat, (Weights_hidden, bias_hidden, Weights_log, bias_log) = climin.util.empty_with_views(tmpl)
    climin.initialize.randomize_normal(wrt_flat, 0, 0.01)
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
        W_hidden = theano.shared(value=Weights_hidden, name='Wh', borrow=True),
        b_hidden = theano.shared(value=bias_hidden, name='bh', borrow=True),
        W_log = theano.shared(value=Weights_log, name='Wo', borrow=True),
        b_log = theano.shared(value=bias_log, name='bo', borrow=True)
    )

    L1_reg = 0.001
    L2_reg = 0.0001
    # cost_reg = (classifier.negative_log_likelihood(y)
                  # + L1_reg * classifier.L1
                  # + L2_reg * classifier.L2_sqr
    # )
    cost_reg = (classifier.errors(y)
               # + L1_reg * classifier.L1
               # + L2_reg * classifier.L2_sqr
    )

    loss = theano.function(
        inputs = [x, y],
        outputs = cost_reg,
        allow_input_downcast = True
    )

    gradients = theano.function(
        inputs=[x, y],
        outputs=T.grad(cost_reg, classifier.params),
        allow_input_downcast=True
    )

    def d_loss_wrt_pars(parameters, inputs, targets):
        g_W_h, g_b_h, g_W_l, g_b_l = gradients(inputs, targets)
        return np.concatenate([g_W_h.flatten(), g_b_h, g_W_l.flatten(), g_b_l])

    zero_one_loss = theano.function(
        inputs = [x, y],
        outputs = classifier.errors(y),
        allow_input_downcast = True
    )

    print('... training: hidden layer act func: %s, lin reg optimizer: %s)' % (active_func_name, opt_name))

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

    print('... using logistic regression optimizer: %s' % opt_name)
    if opt_name == 'Bfgs':
        opt = climin.Bfgs(wrt_flat, loss, d_loss_wrt_pars, args=args)
    elif opt_name == 'Lbfgs':
        opt = climin.Lbfgs(wrt_flat, loss, d_loss_wrt_pars, args=args)
    elif opt_name == 'Nlcg':
        opt = climin.NonlinearConjugateGradient(wrt_flat, loss, d_loss_wrt_pars, args=args)
    # elif opt_name == 'GradientDescent':
    #     opt = climin.GradientDescent(wrt_flat, d_loss_wrt_pars, step_rate=0.1, momentum=.95, args=args)
    elif opt_name == 'RmsProp':
        opt = climin.RmsProp(wrt_flat, d_loss_wrt_pars, step_rate=1e-4, decay=0.9, args=args)
    elif opt_name == 'Rprop':
        opt = climin.Rprop(wrt_flat, d_loss_wrt_pars, args=args)
    elif opt_name == 'Adam':
        opt = climin.Adam(wrt_flat, d_loss_wrt_pars, step_rate=0.0002, decay=0.99999999, decay_mom1=0.1,
                          decay_mom2=0.001, momentum=0, offset=1e-08, args=args)
    elif opt_name == 'Adadelta':
        opt = climin.Adadelta(wrt_flat, d_loss_wrt_pars, step_rate=1, decay=0.9, momentum=.95, offset=0.0001, args=args)
    else:
        opt = climin.GradientDescent(wrt_flat, d_loss_wrt_pars, step_rate=1e-11, momentum=.95, args=args)

    for info in opt:
        iter = info['n_iter']
        epoch = iter // n_train_batches

        # stdout.write('\r%.2f%% of Epoch %d' % (float(iter * 100) / n_train_batches - epoch * 100, (epoch + 1)))
        # stdout.flush()

        if (iter + 1) % validation_frequency == 0:
            valid_loss.append(zero_one_loss(valid_set_x, valid_set_y))
            train_loss.append(zero_one_loss(train_set_x, train_set_y))

            this_validation_loss = np.mean(valid_loss)

            print('epoch %i, validation error %f %%' % (epoch, this_validation_loss * 100.))
            logging.info('epoch %i, validation error %f %%' % (epoch, this_validation_loss * 100.))

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                # improve patience if loss improvement is good enough
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
                logging.info(('     epoch %i, test error of'' best model %f %%') %(epoch,test_score * 100.))

        if patience <= iter or epoch >= n_epochs:
            break

    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    logging.info(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    return classifier
