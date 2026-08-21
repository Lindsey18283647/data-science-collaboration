"""Model training utilities."""

from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    """Train a machine learning model.
    
    Use Random Forest with specific parameters
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model