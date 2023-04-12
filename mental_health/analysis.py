import os

from typing import List, Tuple

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
import utils
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (max_error, mean_absolute_error,
                             mean_squared_error, median_absolute_error,
                             r2_score, mean_squared_log_error, d2_absolute_error_score)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor

import math


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

    CMAP = sns.color_palette(n_colors=7).as_hex()
    Y_COLOR = CMAP[-1]
    PALETTE = {
        'LinearRegression': CMAP[0],
        'DecisionTreeRegressor': CMAP[1],
        'MLPRegressor': CMAP[2],
        'XGBRegressor': CMAP[3],
        'RandomForestRegressor': CMAP[4],
        'KNeighborsRegressor': CMAP[5],
        'Y': Y_COLOR,

    }
    ALL_METRICS = [mean_squared_error, utils.root_mean_squared_error,
                   r2_score, mean_absolute_error, max_error, ]

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
        self.predictions_test = predictions
        return predictions

    def evaluate(self, metric=mean_absolute_error, verbose: bool = False) -> dict:
        scores = {}
        for model_name in self.models.keys():
            scores[model_name] = metric(y_true=self.Y_test,
                                        y_pred=self.predictions_test[model_name])
            if verbose:
                print(f"{model_name}: {scores[model_name]}")
        return scores

    def visualize_predictions(self, sample_range: Tuple = (0, 32)) -> None:
        """
        Visualizes the predicitons of all models plotted against the ground truth Y
        by default only a subset of predicitons is plotted (0,32), this range can be adapted with @sample_range
        """
        sr_start, sr_end = sample_range
        preds_subset = {k: preds[sr_start:sr_end]
                        for k, preds in self.predictions_test.items()}
        preds_subset['Y'] = self.Y_test[sr_start:sr_end]

        pred_df = pd.DataFrame(preds_subset)

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_xlabel('Datapoint index')
        ax.set_ylabel('Number of suicides (Y and model predicitons)')
        sns.scatterplot(pred_df, markers=True, alpha=.6,
                        ax=ax, palette=self.PALETTE)
        sns.lineplot(pred_df.Y, alpha=.9, ax=ax,
                     color=self.Y_COLOR, linewidth=2, legend=False)
        sns.lineplot(pred_df[[k for k in self.PALETTE.keys() if k != 'Y']],
                     alpha=.5, ax=ax, palette=self.PALETTE, linewidth=.8, legend=False)

    def visualize_metrics(self, metrics: List = ALL_METRICS) -> None:
        num_metrcis = len(metrics)
        ncols = 2
        nrows = math.ceil(num_metrcis / 2)
        fig, ax = plt.subplots(figsize=(15, 15), nrows=nrows, ncols=ncols)
        for col in range(ncols):
            for row in range(nrows):
                index = row*ncols + col
                if index >= num_metrcis:
                    break
                metric = metrics[index]
                df = pd.DataFrame({
                    'model': self.models.keys(),
                    'score': self.evaluate(metric=metric).values(),
                })
                sns.barplot(df, x='model', y='score',
                            ax=ax[row][col], palette=self.PALETTE)
                ax[row][col].set_title(metric.__name__)
