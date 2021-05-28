import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression


def log_likelihood(x, y, sample_size):
    reg = LinearRegression().fit(x, y)
    ssr = np.sum((y - reg.predict(x)) ** 2)
    half_size = sample_size / 2
    log_lik = -half_size * np.log(2 * np.pi) - half_size * np.log(ssr / sample_size) - half_size
    return log_lik, reg


def likelihood_ratio_test(alternate_log_lik, null_log_lik, degrees_of_freedom):
    lr = 2 * (alternate_log_lik - null_log_lik)
    p = chi2.sf(lr, degrees_of_freedom)
    return lr, p


def run_likelihood_ratio_test(alternate_x, null_x, y):
    sample_size = len(y)
    alternate_log_lik = log_likelihood(alternate_x, y, sample_size)[0]
    null_log_lik = log_likelihood(null_x, y, sample_size)[0]
    degrees_of_freedom = alternate_x.shape[1] - null_x.shape[1]
    lr, p = likelihood_ratio_test(alternate_log_lik, null_log_lik, degrees_of_freedom)

    print(f"Log-likelihood of alternate (full) model = {alternate_log_lik}")
    print(f"Log-likelihood of null model = {null_log_lik}")
    print(f"likelihood ratio = {lr}, p = {p} with {degrees_of_freedom} degrees of freedom\n")


def hierarchical_lrt(x, y, names_x, significance_value=0.05, indentation_level=0):
    num_variables = x.shape[1]
    extra_space = "    " * indentation_level
    print(f"{extra_space}Analyzing model with {names_x}...\n")
    sample_size = len(y)
    alternate_log_lik, reg = log_likelihood(x, y, sample_size)
    total_r_squared = reg.score(x, y)
    print(f"{extra_space}    explained variance = {total_r_squared:.2f}\n")

    for variable_num in range(num_variables):
        indices = np.array([i for i in range(num_variables) if i != variable_num])
        null_log_lik, reg = log_likelihood(x[:, indices], y, sample_size)
        r_squared = reg.score(x[:, indices], y)
        null_names = [name for name_num, name in enumerate(names_x) if name_num != variable_num]
        print(f"{extra_space}    Assessing reduced model with {null_names}...")
        lr, p = likelihood_ratio_test(alternate_log_lik, null_log_lik, degrees_of_freedom=1)
        print(f"{extra_space}    p = {p:.2e}, model reduction{' NOT ' if p < significance_value else ' '}justified")
        print(f"{extra_space}    variance uniquely explained by '{names_x[variable_num]}' = "
              f"{total_r_squared - r_squared:.2f}\n")

        if len(indices) > 1:
            hierarchical_lrt(x[:, indices], y, null_names, indentation_level=indentation_level + 2)

