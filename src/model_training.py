"""Model training utilities."""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_model(X_train, y_train, model_type='random_forest'):
    """Train a machine learning model."""
    if model_type == 'gradient_boosting':
        # Use Gradient Boosting for better performance
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    else:
        # Use Random Forest as default
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,   # 这里保留同事修改的值
            random_state=42
        )
    model.fit(X_train, y_train)
    return model