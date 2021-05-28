import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from likelihood_ratio_test import run_likelihood_ratio_test, hierarchical_lrt


# TRY OUT THE LIKELIHOOD RATIO TEST

# Generate some data
sample_size = 200

xs = []
xs.append(np.random.normal(10, 2, sample_size))
xs.append(np.random.normal(30, 10, sample_size))
xs.append(np.random.normal(20, 5, sample_size))
X = sm.add_constant(np.column_stack(xs))

beta = [2,
        1,
        -0.5,
        0.01]

y = np.dot(X, beta) + np.random.normal(0, 1, sample_size)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xs[0], xs[1], y)


# My implementation with sklearn
run_likelihood_ratio_test(X[:, 1:], X[:, (1, 2)], y)
hierarchical_lrt(X[:, 1:], y, ["x1", "x2", "x3"])


# Statmodels' methods
print("STATSMODELS...")
model = sm.OLS(y, X)
results = model.fit()
print("Alternative (full) model fit:")
print(results.summary())

print("Null model fit:")
null_model = sm.OLS(y, X[:, (0, 1, 2)])
null_results = null_model.fit()
print(null_results.summary())

lr_results = results.compare_lr_test(null_results)
print(f"\nlikelihood ratio = {lr_results[0]}, p = {lr_results[1]}, degrees of freedom = {lr_results[2]}")

# plt.show()
