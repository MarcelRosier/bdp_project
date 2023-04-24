import math
import os
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
import torch
import utils
from codecarbon import EmissionsTracker
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (d2_absolute_error_score, max_error,
                             mean_absolute_error, mean_squared_error,
                             mean_squared_log_error, median_absolute_error,
                             r2_score)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor


class ModelAnalysis:

    ALL_MODELS_UNTRAINED = {
        'LinearRegression': LinearRegression(positive=False),
        'DecisionTreeRegressor': DecisionTreeRegressor(random_state=0),
        # 'MLPRegressor': MLPRegressor(
        #     hidden_layer_sizes=[512, 256, 64, 8],
        #     max_iter=3000,
        #     activation='relu'),
        'XGBRegressor': XGBRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'KNeighborsRegressor': KNeighborsRegressor()
    }

    CMAP = sns.color_palette(n_colors=7).as_hex()
    Y_COLOR = CMAP[-1]
    PALETTE = {
        'LinearRegression': CMAP[0],
        'DecisionTreeRegressor': CMAP[1],
        'NN': CMAP[2],
        'XGBRegressor': CMAP[3],
        'RandomForestRegressor': CMAP[4],
        'KNeighborsRegressor': CMAP[5],
        'Y': Y_COLOR,

    }
    ALL_METRICS = [mean_squared_error, utils.root_mean_squared_error,
                   r2_score, mean_absolute_error, max_error, ]

    def __init__(self, models: dict, y: str = 'suicides_per_100k_pop') -> None:
        self.models: dict = models
        # self.X_train_std, self.X_test_std, self.Y_train, self.Y_test = utils.get_train_test_split()
        self.splits = utils.get_train_val_test_split(y=y)

        # self.X_train, self.y_train = self.splits['train']['X'], self.splits['train']['y']
        # self.X_val, self.y_val = self.splits['val']['X'], self.splits['val']['y']
        # self.X_test, self.y_test = self.splits['test']['X'], self.splits['test']['y']

        self.predictions = {}

    def train(self, track_emissions: bool = False) -> None:
        for model_name, model in self.models.items():
            if track_emissions:
                tracker = EmissionsTracker(project_name=model_name)
                tracker.start()
            print(f"Fitting {model_name}")
            model.fit(self.splits['train']['X'], self.splits['train']['y'])
            if track_emissions:
                tracker.stop()

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

    def predict(self, split: str = 'test') -> dict:
        predictions = {}
        for model_name, model in self.models.items():
            X = self.splits[split]['X']
            if model_name == "NN":
                with torch.no_grad():
                    model.eval()
                    pred = model(torch.tensor(X).float())
                    predictions[model_name] = pred[:, 0]
            else:
                predictions[model_name] = model.predict(X)
        self.predictions[split] = predictions
        return predictions

    def evaluate(self, metric=utils.root_mean_squared_error, split: str = 'test', verbose: bool = False) -> dict:
        scores = {}
        for model_name in self.models.keys():
            scores[model_name] = metric(
                y_true=self.splits[split]['y'],
                y_pred=self.predictions[split][model_name]
            )
            if verbose:
                print(f"{model_name}: {scores[model_name]}")
        order = max if metric == r2_score else min
        best = order(scores, key=scores.get)
        return scores, best

    def visualize_carbon(self):
        df = pd.read_csv('emissions.csv')
        df = df[df.project_name.isin(
            ['linear_default', 'tree_md_18_ms_4', 'forest_ne_30_md_18_ms_2', 'MLP'])]
        df = df.replace({
            'linear_default': 'LinearRegression',
            'tree_md_18_ms_4': 'DecisionTreeRegressor',
            'forest_ne_30_md_18_ms_2': 'RandomForestRegressor',
            'MLP': 'NN'
        })
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(df, x='project_name', y='emissions',
                    log=True, palette=self.PALETTE, ax=ax)
        # ax.legend(labels=df.project_name.values)
        ax.set_ylabel('emissions in $CO_2eq$')
        ax.set_xlabel(None)

    def visualize_predictions(self, sample_range: Tuple = (0, 32), split: str = 'test', palette=sns.color_palette(), sort: bool = False) -> None:
        """
        Visualizes the predicitons of all models plotted against the ground truth Y
        by default only a subset of predicitons is plotted (0,32), this range can be adapted with @sample_range
        """
        sr_start, sr_end = sample_range
        if sort:
            pred_keys = self.predictions[split].keys()
            all_preds = self.predictions[split].values()
            y = self.splits[split]['y']
            sorted_preds = sorted(zip(y, *all_preds))
            y_sorted, *all_preds_sorted = zip(*sorted_preds)
            preds_subset = {k: preds[sr_start:sr_end]
                            for k, preds in zip(pred_keys, all_preds_sorted)}
            # map tensor to float
            preds_subset['NN'] = list(
                map(lambda x: x.item(), preds_subset['NN']))
            # print(preds_subset['NN'])
            preds_subset['Y'] = y_sorted[sr_start:sr_end]
        else:
            preds_subset = {k: preds[sr_start:sr_end]
                            for k, preds in self.predictions[split].items()}
            preds_subset['Y'] = self.splits[split]['y'][sr_start:sr_end]

        pred_df = pd.DataFrame(preds_subset)

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_xlabel('Datapoint index')
        ax.set_ylabel(
            'Number of suicides per 100k population (Y and model predictions)')
        sns.scatterplot(pred_df, markers=True, alpha=.6,
                        ax=ax, palette=palette)
        sns.lineplot(pred_df.Y, alpha=.9, ax=ax,
                     color=self.Y_COLOR, linewidth=2, legend=False)
        sns.lineplot(pred_df[[k for k in self.models.keys() if k != 'Y']],
                     alpha=.5, ax=ax, palette=palette, linewidth=.8, legend=False)
        ax.axhline(c='black', alpha=.1)

    def visualize_metrics(self, metrics: List = ALL_METRICS, split: str = 'test', palette=None, verbose=False, ncols: int = 2) -> None:
        num_metrcis = len(metrics)
        nrows = math.ceil(num_metrcis / ncols)
        fig, ax = plt.subplots(figsize=(15, 5), nrows=nrows, ncols=ncols)
        for col in range(ncols):
            for row in range(nrows):
                index = row*ncols + col
                if index >= num_metrcis:
                    break
                metric = metrics[index]
                scores, min_score = self.evaluate(
                    metric=metric, split=split)
                df = pd.DataFrame({
                    'model': self.models.keys(),
                    'score': scores.values(),
                })
                if verbose:
                    print(metric.__name__)
                    print(f"min: {min_score}")
                    print(df)
                cax = cax = ax[col]  # ax[row][col]
                sns.barplot(df, x='model', y='score', ax=cax, palette=palette)
                cax.set_title(metric.__name__, loc='left')
                cax.set_xlabel(None)
                if col > 0:
                    cax.set_ylabel(None)
                if metric == r2_score:
                    cax.set_ylim((0, 1))
                cax.set_xticklabels([])
        patches = [mpatches.Patch(
            color=self.PALETTE[m],
            label=m)
            for m in self.models.keys()]
        fig.legend(handles=patches)  # , loc='center right')
