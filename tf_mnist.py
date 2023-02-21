import tensorflow as tf
from keras import backend
import numpy as np
from numpy.linalg import norm
from scipy.optimize import fsolve
import gc
import multiprocessing as mp

np.random.seed(0)

class L1Constraint(tf.keras.constraints.Constraint):
    def __init__(self, max_value, axis=0):
        self.max_value = max_value
        self.axis = axis

    def __call__(self, w):
        norms = tf.reduce_sum(tf.abs(w), axis=self.axis, keepdims=True)

        desired = backend.clip(norms, 0, self.max_value)
        return w * (desired / (backend.epsilon() + norms))

"""
Trains a single-layer feed-forward neural network
x -> sigmoid(w * sigmoid(U x))
where sigmoid is applied componentwise, * is the dot product, and ||w||_1 <= t. All biases set to 0 for simplicity.
"""
def train_l1_penalty(xs, ys, t):
    U_constraint = u_matrix_upper_bound
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(5, activation='sigmoid', kernel_constraint=tf.keras.constraints.MaxNorm(U_constraint, axis=1), bias_constraint=tf.keras.constraints.MaxNorm(U_constraint, axis=0)),
      tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_constraint=L1Constraint(t), bias_constraint=tf.keras.constraints.MaxNorm(0, axis=0))
    ])
    model.compile(optimizer='nadam', loss=tf.keras.losses.MeanAbsoluteError())
    model.fit(xs, ys, epochs = 20, verbose=0) # Maybe set to 30 epochs?

    risk = model.evaluate(xs, ys, verbose=0)
    gc.collect()

    return risk

def epsilon(x_train, alpha):
    emp_rademacher_complexity = 2 * u_matrix_upper_bound * weight_upper_bound/len(x_train)*(sum([norm(x.flatten())**2 for x in x_train]))**0.5
    return 2 * emp_rademacher_complexity + 3*(np.log(4/alpha)/(2*len(x_train)))**0.5

def boostrapped_sample(x_train, y_train):
    bootstrapped_indices = np.random.choice(len(x_train), len(x_train))

    return x_train[bootstrapped_indices], y_train[bootstrapped_indices]

"""
Set up the MNIST data set, with labels for even (0) and odd (1) examples
"""
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 2**17, x_test / 2**17

y_train = y_train % 2
y_test = y_test % 2


alpha = .207274
gamma = alpha - .0000001
u_matrix_upper_bound = 18
weight_upper_bound = 5
hypothesized_upper_bound = 0.25

B = 50

print("Type I Error Rate: ", alpha + np.exp(-6*B*gamma**2/(4*gamma+3)))

pl_boot = 0
for i in range(B):
    print("Iteration:", i)
    boot_x, boot_y = boostrapped_sample(x_train, y_train)

    eps = epsilon(boot_x, alpha)

    erm_risk = train_l1_penalty(boot_x, boot_y, weight_upper_bound)

    constrained_risk = train_l1_penalty(x_train, y_train, hypothesized_upper_bound)

    print(constrained_risk, erm_risk, constrained_risk - erm_risk, 2*eps)
    if constrained_risk - erm_risk < 2*epsilon(boot_x, alpha):
        pl_boot += 1
    print("Plausibility:", pl_boot/(i+1))

pl_boot /= B
print("Final plausibility:", pl_boot)
print("Critical value:", 1 - alpha - gamma)
