import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

homes = pd.read_csv('real_estate.csv')
# Distance in original dataset is metres; divide by 100 for interpretable slope.

homes['distance_100m'] = homes['X3 distance to the nearest MRT station'] / 100
X = homes[['distance_100m']] # Double brackets => DataFrame (2D).
y = homes['Y house price of unit area']

model = LinearRegression()
model.fit(X, y) # Learn slope and intercept from all 414 observations.
slope = model.coef_[0]
intercept = model.intercept_
print('Rows:', len(homes))
print(f'Slope per additional 100 m: {slope:.3f}')
print(f'Intercept: {intercept:.3f}')

# 500 metres = 5 units of 100 metres. Feature name must match training.
query = pd.DataFrame({'distance_100m': [5]})
print(f'Predicted price at 500 m: {model.predict(query)[0]:.2f}')
print('Interpretation: each additional 100 m changes predicted price by', round(slope, 3), 'price units.')
# Sort x so the fitted line draws smoothly from left to right.
order = X['distance_100m'].argsort()
plt.scatter(X['distance_100m'], y, alpha=.4, label='Observed homes')
plt.plot(X['distance_100m'].iloc[order], model.predict(X).take(order), label='Fitted line')
plt.xlabel('Distance to nearest MRT (100 m units)')
plt.ylabel('Price (10,000 NTD per ping)')
plt.title('Housing price vs MRT distance')
plt.legend()
plt.tight_layout()
plt.savefig('task1_housing.png', dpi=160)
plt.close()
