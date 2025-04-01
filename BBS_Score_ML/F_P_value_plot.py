import matplotlib.pyplot as plt
import numpy as np
import scipy

f_p_values = scipy.io.loadmat(".../f_p_values.mat");

f_values = f_p_values['f_values'];
p_values = f_p_values['p_values'];

labels = ['Diff of Stance Time', 'Foot Freq', 'Chest Freq', 'Step Number per Min', 
          'IDS', 'Foot Amp', 'Chest Amp', 'SMTR', 'Gait Speed', 'Stride Length']


"""
P-value plot
"""
log_data = np.abs(np.log10(p_values))
sorted_data = np.sort(log_data)
threshold = np.abs(np.log10(0.05))

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(np.squeeze(sorted_data))), np.squeeze(sorted_data), color='lightblue', edgecolor='black')
ax.axhline(y=threshold, color='red', linestyle='--')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
ax.set_yticklabels(ax.get_yticks(), fontsize=14)
ax.set_ylabel('|log10(Pvalue)|', fontsize=16)

for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.show()

"""
F-value plot
"""
sorted_f_values = np.sort(f_values)
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(np.squeeze(sorted_f_values))), np.squeeze(sorted_f_values), color='lightblue', edgecolor='black')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
ax.set_ylabel('F-value', fontsize=16)
for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.show()
