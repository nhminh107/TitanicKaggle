from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Model: 
    def __init__(self, isRunning: bool):
        self.model = RandomForestClassifier(random_state=42)
        self.isRunning = isRunning
        self.best_model = None

    def fit(self, X_train, y_train):
        if self.isRunning:
            return None

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
        self.isRunning = True

    def predict(self, X_test):
        if not self.isRunning:
            return None

        y_pred = self.best_model.predict(X=X_test)
        return y_pred

