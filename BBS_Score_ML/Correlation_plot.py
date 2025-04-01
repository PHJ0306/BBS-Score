import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(".../10Feature+BBS_Score(+IDS).csv")
data = data.drop(columns='NUM')

corr_matrix = data.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
lower_triangle = corr_matrix.copy()
lower_triangle[mask] = np.nan
lower_triangle_df = pd.DataFrame(lower_triangle, index=corr_matrix.index, columns=corr_matrix.columns)

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(lower_triangle_df, annot=True, annot_kws={'color': 'white'}, cmap='Greys', vmin=-1, vmax=1, cbar=False, ax=ax)

for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2)

plt.show()
