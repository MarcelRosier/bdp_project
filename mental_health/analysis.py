import os

from typing import Tuple

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
import utils
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (max_error, mean_absolute_error,
                             mean_squared_error, median_absolute_error,
                             r2_score)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor


class ModelAnalysis:

    ALL_MODELS_UNTRAINED = {
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

    def save_models(self, base_path: str = "models") -> None:
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        for model_name, model in self.models.items():
            dump(model, f"{base_path}/{model_name}.joblib")

    def load_models(self, base_path: str = "models") -> None:
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

    def visualize(self, sample_range: Tuple = (0, 32)):
        """
        Visualizes the predicitons of all models plotted against the ground truth Y
        by default only a subset of predicitons is plotted (0,32), this range can be adapted with @sample_range
        """
        sample_range_start, sample_range_end = sample_range
        preds_subset = {k: preds[sample_range_start:sample_range_end]
                        for k, preds in self.predicitons_test.items()}
        preds_subset['Y'] = self.Y_test[sample_range_start:sample_range_end]

        pred_df = pd.DataFrame(preds_subset)
        cmap = sns.color_palette(n_colors=7).as_hex()

        Y_COLOR = cmap[-1]
        palette = {
            'LinearRegression': cmap[0],  # 'b',
            'DecisionTreeRegressor': cmap[1],  # 'black',
            'MLPRegressor': cmap[2],  # 'r',
            'XGBRegressor': cmap[3],  # 'c',
            'RandomForestRegressor': cmap[4],  # 'y',
            'KNeighborsRegressor': cmap[5],  # 'k',
            'Y': Y_COLOR,  # 'g'

        }
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_xlabel('Datapoint index')
        ax.set_ylabel('Number of suicides (Y and model predicitons)')
        sns.scatterplot(pred_df, markers=True, alpha=.6,
                        ax=ax, palette=palette)
        sns.lineplot(pred_df.Y, alpha=.9, ax=ax,
                     color=Y_COLOR, linewidth=2, legend=False)
        sns.lineplot(pred_df[[k for k in palette.keys() if k != 'Y']],
                     alpha=.5, ax=ax, palette=palette, linewidth=.8, legend=False)
