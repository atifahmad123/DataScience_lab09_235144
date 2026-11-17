import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

concrete = pd.read_csv('concrete_strength.csv').drop_duplicates()
target = 'Concrete compressive strength'
X = concrete.drop(columns=target) # All 8 input columns (7 ingredients + curing age).
y = concrete[target]             # Strength in MPa is what we predict.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
print('Rows after deduplication:', len(concrete))
print('Train/test:', len(X_train), len(X_test))
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = 'neg_root_mean_squared_error'
# Baseline uses training folds only; do not inspect test set yet.
baseline_errors = -cross_val_score(LinearRegression(), X_train, y_train, cv=folds, scoring=scoring)
print(f'Linear baseline mean CV RMSE: {baseline_errors.mean():.2f} MPa')

pipeline = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scale', StandardScaler()), # Refit scaler separately inside each training fold.
    ('ridge', Ridge()),
])
params = {'poly__degree': [1, 2, 3], 'ridge__alpha': [.01, .1, 1, 10, 100]}
search = GridSearchCV(pipeline, params, cv=folds, scoring=scoring, refit=True)
search.fit(X_train, y_train) # Tries 15 settings * 5 folds = 75 CV fits (+ refit).
cv_results = pd.DataFrame(search.cv_results_)
top3 = cv_results.sort_values(['rank_test_score', 'mean_test_score'], ascending=[True, False]).head(3)
print('\nTop 3 settings:')
for _, row in top3.iterrows():
    print('Rank:', int(row['rank_test_score']), 'Parameters:', row['params'],
          'Mean CV RMSE:', round(-row['mean_test_score'], 2))
print('Best parameters:', search.best_params_)
print(f'Tuned mean CV RMSE: {-search.best_score_:.2f} MPa')
print(f'CV improvement: {baseline_errors.mean() + search.best_score_:.2f} MPa')

pred = search.predict(X_test) # Held-out evaluation, once, after tuning.
print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f} MPa')
print(f'Test MAE: {mean_absolute_error(y_test, pred):.2f} MPa')
low = min(y_test.min(), pred.min())
high = max(y_test.max(), pred.max())
plt.scatter(y_test, pred, alpha=.6, label='Predictions')
plt.plot([low, high], [low, high], '--', label='Perfect prediction (y=x)')
plt.xlabel('Actual strength (MPa)')
plt.ylabel('Predicted strength (MPa)')
plt.title('Concrete: actual vs predicted strength')
plt.legend()
plt.tight_layout()
plt.savefig('task3_concrete.png', dpi=160)
plt.close()
