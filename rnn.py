import numpy as np
from utils import *
import operator

class RNN:

    def __init__(self, t_dim, hidden_dim=100, truncate=4):
        # Assign instance variables
        self.t_dim = t_dim
        self.hidden_dim = hidden_dim
        self.truncate = truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1. / t_dim), np.sqrt(1. / t_dim), (hidden_dim, t_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (t_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        T = len(x)
        h = np.zeros((T + 1, self.hidden_dim))
        h[-1] = np.zeros(self.hidden_dim)
        yhat = np.zeros((T, self.t_dim))
        for t in np.arange(T):
            h[t] = np.tanh(self.U[:,x[t]] + self.W.dot(h[t-1]))
            yhat[t] = softmax(self.V.dot(h[t]))
        return [yhat, h]

    def predict(self, x):
        yhat, h = self.forward_propagation(x)
        return np.argmax(yhat, axis=1)

    def calculate_total_loss(self, xx, yy):
        Loss = 0
        for i in np.arange(len(yy)):
            yhat, h = self.forward_propagation(xx[i])
            correct_t_predictions = yhat[np.arange(len(yy[i])), yy[i]]
            Loss += -1 * np.sum(np.log(correct_t_predictions))
        return Loss

    def calculate_loss(self, xx, yy):
        N = np.sum((len(y_list) for y_list in yy))
        return self.calculate_total_loss(xx, yy) / N

    def bptt(self, x, y):
        T = len(y)
        yhat, h = self.forward_propagation(x)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)
        dU = np.zeros(self.U.shape)
        diff = yhat
        diff[np.arange(len(y)), y] -= 1
        for t in np.arange(T)[::-1]:
            dV += np.outer(diff[t], h[t].T)
            delta = self.V.T.dot(diff[t]) * (1 - (h[t] ** 2))
            for k in np.arange(max(0, t-self.truncate), t+1)[::-1]:
                dW += np.outer(delta, h[k-1])
                dU[:,x[k]] += delta
                delta = self.W.T.dot(delta) * (1 - h[k-1] ** 2)
        return [dV, dW, dU]

    def sgd_step(self, x, y, learning_rate):
        dV, dW, dU = self.bptt(x, y)
        self.V -= learning_rate * dV
        self.W -= learning_rate * dW
        self.U -= learning_rate * dU

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['V', 'W', 'U']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                        np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)




