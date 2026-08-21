"""Model training utilities."""

from sklearn.ensemble import GradientBoostingClassifier

def train_model(X_train, y_train):
    """Train a machine learning model.
    
    Use Gradient Boosting for better performance
    """
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model