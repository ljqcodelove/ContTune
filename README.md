# ContTune

ContTune, a continuous tuning system for elastic stream processing using Big-small algorithm and conservative Bayesian Optimization (CBO) algorithm.

ContTune is simple and useful! And we faithfully recommend you to read DS2[^1].

ContTune has been deployed on Tencent's distributed stream data processing system _Oceanus_ and serves as the only parallelism tuning system.  The codes of Section **Implementation** (containing **controller** and **Databases**) is not open source, but these codes are only engineering realization, and we give our environment codes of Big-small algorithm and conservative Bayesian Optimization (CBO). 

## Requirements

- Any version of Apache Flink
- Python >= 3.6
- Python-scipy >= 1.5.2
- Python-matplotlib >= 3.3.4

## Quick Start

### Historical Observations Preparation

Historical observations contain pairs of (parallelism, processing ability), where parallelism can be obtained on Flink and processing ability can be obtained on Flink after deploying **Codes of Getting Metrics**.

Here is historical observations of Nexmark Q1 operator: 'Mapper' from database:

| Parallelism | 1st PA      | 2nd PA      | 3rd PA      | 4th PA      | 5th PA      | mean-reversion PA                                            |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ------------------------------------------------------------ |
| 1           | 537531.4221 | 526027.042  | 536474.7595 | 539867.5141 | 507965.0999 | (537531.4221 + 526027.042 + 536474.7595 + 539867.5141 + 507965.0999) / 5 = 529573.16752 |
| 2           | 972028.3529 | 971570.0773 | 953960.6134 | 978189.2728 | 958338.4957 | (972028.3529 + 971570.0773 + 953960.6134 + 978189.2728 + 958338.4957) / 5 = 966817.36242 |
| 3           |             |             |             |             |             |                                                              |
| 4           | 1836718.786 | 1804663.382 | 1850355.944 | 1839823.777 | 1816916.011 | (1836718.786 + 1804663.382 + 1850355.944 + 1839823.777 + 1816916.011) / 5 = 1829695.58 |
| 5           |             |             |             |             |             |                                                              |
| 6           | 2541894.169 | 2509173.876 | 2559861.549 | 2488776.34  | 2577807.874 | (2541894.169 + 2509173.876 + 2559861.549 + 2488776.34 + 2577807.874) / 5 = 2535502.7616 |
| 7           | 2933310.123 | 2975408.126 | 2958610.669 | 2898671.852 | 2918442.125 | (2933310.123 + 2975408.126 + 2958610.669 + 2898671.852 + 2918442.125) / 5 = 2936888.579 |
| 8           | 3118477.883 | 3101777.888 | 3147521.943 | 3108545.88  | 3116065.819 | (3118477.883 + 3101777.888 + 3147521.943 + 3108545.88 + 3116065.819) / 5 = 3118477.8826 |
| 9           |             |             |             |             |             |                                                              |
| 10          | 3713964.595 | 3760726.143 | 3798472.095 | 3711569.576 | 3620763.509 | (3713964.595 + 3760726.143 + 3798472.095 + 3711569.576 + 3620763.509) / 5 = 3721099.1836 |
| 11          |             |             |             |             |             |                                                              |
| 12          |             |             |             |             |             |                                                              |
| 13          |             |             |             |             |             |                                                              |
| 14          | 4624714.166 | 4585548.142 | 4663880.189 |             |             | (4624714.166 + 4585548.142 + 4663880.189) / 3 = 4624714.165666 |
| 15          | 4936270.001 | 5026367.858 | 4982989.464 | 4913003.223 | 4980510.747 | (4936270.001 + 5026367.858 + 4982989.464 + 4913003.223 + 4980510.747) / 5 = 4967828.2586 |
| 16          |             |             |             |             |             |                                                              |
| 17          |             |             |             |             |             |                                                              |
| 18          |             |             |             |             |             |                                                              |
| 19          |             |             |             |             |             |                                                              |
| 20          | 6309578.518 | 6257840.038 | 6361316.999 | 6379096.212 | 6445553.46  | (6309578.518 + 6257840.038 + 6361316.999 + 6379096.212 + 6445553.46) / 5 = 6350677.0454 |
| 21          | 6465249.805 | 6531482.256 | 6461423.401 | 6358448.484 | 6509645.079 | (6465249.805 + 6531482.256 + 6461423.401 + 6358448.484 + 6509645.079) / 5 = 6465249.805 |
| 22          |             |             |             |             |             |                                                              |
| 23          |             |             |             |             |             |                                                              |
| 24          |             |             |             |             |             |                                                              |
| 25          | 7071566.573 | 7108160.82  | 7108160.82  | 7068030.475 | 7091017.849 | (7071566.573 + 7108160.82 + 7108160.82 + 7068030.475 + 7091017.849) / 5 = 7089387.3074 |
| 26          | 7476391.84  |             |             |             |             | (7476391.84) / 1 = 7476391.84                                |

From historical observations we can get this map (key is parallelism and value is processing ability):

```python
mp = {}
mp[1] = 529573.16752
mp[2] = 966817.36242
mp[4] = 1829695.58
mp[6] = 2535502.7616
mp[7] = 2936888.579
mp[8] = 3118477.8826
mp[10] = 3721099.1836
mp[13] = 4624714.165666
mp[15] = 4967828.2586
mp[20] = 6350677.0454
mp[21] = 6465249.805
mp[25] = 7089387.3074
mp[26] = 7476391.84
```

And for the tuned job, ContTune first uses the Big-small algorithm to make it non-backpressure, and get the surrogate model like this:

![image-20230301215919441](.\figures\readme1.png)

And now for any upstream rate, you can use the acquisition function to find the best parallelism by ContTune.

For example, if the upstream rate now is 4e6, and you can get the parallelism is:

![image-20230301215919441](.\figures\readme2.png)
$$
\lceil 10.64 \rceil = 11, and \ d_{nearest} = \lvert 11 - 10 \rvert = 1 \ for \ any \ \alpha \geq 1, \ paralllism = 11 \ is \ recommended, \ otherwise, \ DS2 \ is \ triggered.
$$
And ContTune works iteratively for each operator. 

## Codes of Getting Metrics

About useful time if you use the Flink version < 1.13, you can add the code to get busyTime on serialization, processing and deserialization like DS2[^1] as the patch in:

```
.\patch\ds2.patch
```

, otherwise, if you use the Flink version >= 1.13, you can use busyTimeMsPerSecond in [Flink Metrics](https://nightlies.apache.org/flink/flink-docs-release-1.13/docs/ops/metrics/ ) as the useful time.

Because, ContTune first uses Big-small to make job non-backpressure, so ContTune uses numRecordsInPerSecond in [Flink Mertics](https://nightlies.apache.org/flink/flink-docs-release-1.13/docs/ops/metrics/ ) of non-backpressure jobs.

ContTune gets the processing ability is:
$$
\frac{numRecordsInPerSecond}{useful time}.
$$
But if you have deployed the DS2[^1], by running DS2[^1] you will get the real processing ability of operator like this:

![image-20230301215919441](.\figures\readme3.png)

, and you can use the processing ability given by DS2, too.

## Codes of Big-small Algorithm

The Big-small algorithm code is very simple and is the same as Algorithm 1 in the paper. 

## Codes of Conservative Bayesian Optimization

The code of our Gaussian Process is in:

```
.\code\GOofContTune.py
```

The code is simple and useful:

```python
from matplotlib import pyplot as plt
from scipy.optimize import minimize

import numpy as np


class GPR:

    # optimize = True means that the GP model need to use hyper parameters optimization: negative_log_likelihood_loss
    # There are two hyper parameters, l and sigma_f
    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 4.37, "sigma_f": 1000000}
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
                   bounds=((1e-5, 1e5), (1e-5, 5e5)),
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

    # you need to change 27 to the upper bound parallelism of this operator
    test_X = np.arange(0, 27, 1).reshape(-1, 1)

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
    plt.show()

```

For using it, you only need to prepare historical observations mp.

## Codes of Benchmark

All codes of Benchmark are as the same as DS2[^1]:

```
.\benchmark\*.java
.\benchmark\wordcount\*.java
```



## Reference

[^1]:https://github.com/strymon-system/ds2
