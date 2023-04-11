import os
import pickle

import utils
from joblib import dump, load
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (max_error, mean_absolute_error,
                             mean_squared_error, median_absolute_error)
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from torch import nn
from xgboost.sklearn import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


class SuicideRegressor(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.model(x)


class SuicideRegressorBN(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_features=in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=8),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.model(x)


class Wrapper:

    ALL_MODELS = {
        'LinearRegression': LinearRegression(positive=False),
        'DecisionTreeRegressor': DecisionTreeRegressor(random_state=0),
        'MLPRegressor': MLPRegressor(
            hidden_layer_sizes=[512, 256, 64, 8],
            max_iter=3000,
            activation='relu'),
        'XGBRegressor': XGBRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'KNeighborsRegressor': KNeighborsRegressor()

    }

    def __init__(self, models: dict) -> None:
        self.models: dict = models
        self.X_train_std, self.X_test_std, self.Y_train, self.Y_test = utils.get_train_test_split()

    def train(self) -> None:
        for model_name, model in self.models.items():
            print(f"Fitting {model_name}")
            model.fit(self.X_train_std, self.Y_train)

    def get_models(self) -> dict:
        return self.models

    def save_models(self, base_path="models") -> None:
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        for model_name, model in self.models.items():
            dump(model, f"{base_path}/{model_name}.joblib")

    def load_models(self, base_path="models") -> None:
        for model_name in self.models.keys():
            self.models[model_name] = load(f"{base_path}/{model_name}.joblib")

    def predict_test(self) -> dict:
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(self.X_test_std)
        self.predicitons_test = predictions
        return predictions

    def evaluate(self, metric=mean_absolute_error) -> dict:
        scores = {}
        for model_name in self.models.keys():
            scores[model_name] = metric(y_true=self.Y_test,
                                        y_pred=self.predicitons_test[model_name])
            print(f"{model_name}: {scores[model_name]}")
        return scores
