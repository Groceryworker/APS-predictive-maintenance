import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, MissingIndicator

def preprocess(df, fit=True, imputer=None, indicator=None):
    df = df.copy()
    
    y = df["class"]
    X = df.drop(columns="class")
    
    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    
    if fit:
        imputer = SimpleImputer(strategy="median")
        indicator = MissingIndicator(features="missing-only")
        
        X_imputed = imputer.fit_transform(X)
        X_missing = indicator.fit_transform(X)
    else:
        X_imputed = imputer.transform(X)
        X_missing = indicator.transform(X)
    
    X_final = np.hstack([X_imputed, X_missing])
    
    return X_final, y, imputer, indicator
