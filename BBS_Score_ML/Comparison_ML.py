import scipy.io as sio
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data = sio.loadmat(".../predictions_Results.mat")
model_keys = [
    'LinearRegression',
    'Ridge',             
    'RandomForest',      
    'Lasso',             
    'ElasticNet',        
    'DecisionTree',     
    'SVR'               
]

model_labels = [
    'Linear Regression',
    'Ridge',
    'Random Forest',
    'Lasso',
    'ElasticNet',
    'Decision Tree',
    'SVR (Linear)'
]

r2_scores = []

for key in model_keys:
    predictions = data[f"{key}_predictions"].flatten()
    true_values = data[f"{key}_true_values"].flatten()
    
    r2 = r2_score(true_values, predictions)
    r2_scores.append(r2)

fig, ax = plt.subplots(figsize=(12, 5))
bars = plt.bar(model_labels, r2_scores, color='gray')
plt.ylabel('R-squared', fontsize=20)
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}', 
             ha='center', va='bottom', fontsize=16)
    
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.show()
