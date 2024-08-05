import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import optuna

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Data preprocessing
def preprocess_data(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor

# Model selection and hyperparameter tuning
def optimize_model(X, y):
    def objective(trial):
        model_name = trial.suggest_categorical('model', ['RandomForest', 'LogisticRegression'])
        if model_name == 'RandomForest':
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_depth = trial.suggest_int('max_depth', 2, 32)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        else:
            C = trial.suggest_float('C', 0.01, 10.0)
            model = LogisticRegression(C=C, max_iter=1000)
        
        pipeline = Pipeline(steps=[('preprocessor', preprocess_data(X)),
                                   ('classifier', model)])

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        return np.mean(y_pred == y_val)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params

# Model evaluation
def evaluate_model(X, y, best_params):
    if best_params['model'] == 'RandomForest':
        model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
    else:
        model = LogisticRegression(C=best_params['C'], max_iter=1000)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocess_data(X)),
                               ('classifier', model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    data_path = 'data/sample_data.csv'
    df = load_data(data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    best_params = optimize_model(X, y)
    evaluate_model(X, y, best_params)
