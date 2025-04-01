import scipy
import numpy as np
import matplotlib.pyplot as plt

Bootstrap_data= scipy.io.loadmat(".../r2_bootstrap_rf.mat");
Bootstrap = Bootstrap_data['r2_bootstrap'].flatten();
r2_mean = np.mean(Bootstrap)
r2_ci = np.percentile(Bootstrap, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(12,5))
plt.hist(Bootstrap, bins=30, color='black', edgecolor='white')
plt.axvline(r2_mean, color='red', linewidth=4, label=f'Mean: {r2_mean:.2f}')
plt.axvline(r2_ci[0], color='blue', linestyle='--', linewidth=4, label=f'2.5th Percentile: {r2_ci[0]:.2f}')
plt.axvline(r2_ci[1], color='blue', linestyle='--', linewidth=4, label=f'97.5th Percentile: {r2_ci[1]:.2f}')
ax.set_ylabel('Counts')
ax.set_xlabel('R-squared')
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.set_ylim([0, 210])
plt.show()
