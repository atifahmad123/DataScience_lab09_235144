"""Task 2: fair 5-fold CV comparison of linear and quadratic yacht models."""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

yachts = pd.read_csv('yacht_hydrodynamics.data', sep=r'\s+', header=None)
X = yachts[[5]] # Sixth column: Froude number; keep 2D.
y = yachts[6]   # Seventh column: residuary resistance.
models = {
    'Linear': LinearRegression(),
    'Quadratic': Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear', LinearRegression()),
    ]),
}
folds = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, model in models.items():
    # Negative sign restores positive RMSE values for easier interpretation.
    errors = -cross_val_score(model, X, y, cv=folds, scoring='neg_root_mean_squared_error')
    results[name] = errors.mean()
    print(f'{name} fold RMSEs:', errors.round(2))
    print(f'{name} mean RMSE: {errors.mean():.2f}')
print('Better model (lower mean CV RMSE):', min(results, key=results.get))
