import numpy as np
import matplotlib.pyplot as plt
import time


# DEMO FOR A MEASURE OF SPREAD


def old_max_summed_distances(pattern_size):
    sum_distance = 0
    for i in range(pattern_size):
        for j in range(i + 1, pattern_size):
            sum_distance += abs(i - j)
    return sum_distance


# turns out this is not bad
def old_spread(pattern, max_sum_distance):
    total_spread = 0
    for i in range(len(pattern)):
        if pattern[i]:
            for j in range(i + 1, len(pattern)):
                if pattern[j]:
                    total_spread += j - i
    return total_spread / max_sum_distance


def max_summed_distances(pattern_size):
    sum_distance = 0
    for n in range(2, pattern_size + 1):
        sum_distance += (n - 1) * n/2
    return sum_distance + 1


def spread(pattern, max_sum_distance):
    total_spread = 0
    for i in np.nonzero(pattern[:-1])[0]:
        for j in np.nonzero(pattern[i + 1:])[0]:
            total_spread += j + 1
    return total_spread / max_sum_distance


num_patterns = 20
pattern_size = 10

patterns = np.where(np.random.random((num_patterns, pattern_size)) > 0.4, 1, 0)


old_max_sum_d = old_max_summed_distances(pattern_size)
max_sum_d = max_summed_distances(pattern_size)


old_spreads = []
spreads = []
for pattern in patterns:
    start = time.time()
    old_spreads.append(old_spread(pattern, old_max_sum_d))
    print("old: ", time.time() - start)
    start = time.time()
    spreads.append(spread(pattern, max_sum_d))
    print("new: ", time.time() - start)

sorted_indices = np.argsort(old_spreads)
patterns = patterns[sorted_indices]
old_spreads = np.array(old_spreads)[sorted_indices]
spreads = np.array(spreads)[sorted_indices]

fig, ax = plt.subplots(figsize=(4, 8))
ax.pcolor(patterns, cmap='binary', edgecolor='C7', linewidth=1.5)
ax.set_aspect('equal')
ax.set_yticks(np.arange(num_patterns) + 0.5, minor=False)
ax.yaxis.tick_right()
ax.set_yticklabels([f"s = {old_spread:.2f}, {spread:.2f}" for old_spread, spread in zip(old_spreads, spreads)])
ax.set_xticks([], [])
plt.tight_layout()
plt.show()
