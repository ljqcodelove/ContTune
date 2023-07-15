from matplotlib import pyplot as plt
from scipy.optimize import minimize

import numpy as np


class GPR:

    # optimize = True means that the GP model need to use hyper parameters optimization: negative_log_likelihood_loss
    # There are two hyper parameters, l and sigma_f
    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 2.5, "sigma_f": 1000000}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)

        # hyper parameters optimization using Marginal Log-likelihood
        # you also can use GaussianProcessRegressor from scikit-learn
        # and here is example:
        # from sklearn.gaussian_process import GaussianProcessRegressor
        # from sklearn.gaussian_process.kernels import ConstantKernel, RBF
        #
        # # fit GPR
        # kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5,
        #                                                                                      length_scale_bounds=(
        #                                                                                      1e-4, 1e4))
        # gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        # gpr.fit(train_X, train_y)
        # mu, cov = gpr.predict(test_X, return_cov=True)
        # test_y = mu.ravel()
        # uncertainty = 1.96 * np.sqrt(np.diag(cov))
        #
        # # plotting
        # plt.figure()
        # plt.title("l=%.1f sigma_f=%.1f" % (gpr.kernel_.k2.length_scale, gpr.kernel_.k1.constant_value))
        # plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
        # plt.plot(test_X, test_y, label="predict")
        # plt.scatter(train_X, train_y, label="train", c="red", marker="x")
        # plt.legend()
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            return loss.ravel()

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                   bounds=((1e-5, 1e5), (1e-5, 1e5)),
                   method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model does not fit yet.")
            return

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)

        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)

def y(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = x + np.random.normal(0, noise_sigma, size=x.shape)
    return y.tolist()

def GP(mp={}):
    arr = []
    brr = []
    for (key, value) in mp.items():
        arr.append(key)
        brr.append(value)

    train_X = np.array(arr).reshape(-1, 1)
    train_y = (np.array(brr).reshape(-1, 1) + np.random.normal(0, 1e-4, size=np.asarray(train_X).shape)).tolist()

    # you need to change 26 to the maximal bound parallelism of this operator, in ContTune, it is given by the Big-small Algorithm
    test_X = np.arange(0, 26, 1).reshape(-1, 1)

    gpr = GPR()
    gpr.fit(train_X, train_y)
    mu, cov = gpr.predict(test_X)
    test_y = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.figure()
    plt.title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))

    plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
    plt.plot(test_X, test_y, label="predict")
    plt.scatter(train_X, train_y, label="train", c="red", marker="x")
    plt.legend()
    plt.ylabel('Processing ability')
    plt.xlabel('Level of parallelism')
    # plt.savefig("./RepeatedExperiment3.pdf", format='pdf', transparent=True)
    plt.show()
