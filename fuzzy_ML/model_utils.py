from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score
import numpy as np
import pickle


class base_model():

    model_type = None

    def __init__(self, X_train, y_train, X_dev, y_dev, type=None, parameters={} ):
        self.parameters = parameters
        if type is None:
            self.model_type = None
        else:
            self.model_type = type
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.trained_model = None

    def train(self):
        trained_model = self.model_type.fit(self.X_train, self.y_train)
        self.trained_model = trained_model
        return trained_model

    def make_prediction(self, X):

        if self.trained_model is None:
            self.train()

        preds = self.trained_model.predict(X)
        probs = self.trained_model.predict_proba(X)[:, 1]
        return preds, probs

    def get_trained_model(self):

        if self.trained_model is None:
            self.train()
        return self.trained_model

    def evaluate_model(self, plot=False):
        """
        Compare machine learning model to baseline performance.
        Computes statistics and shows ROC curve.
        :param predictions:
        :param probs:
        :param train_predictions:
        :param train_probs:
        :return:
        """

        predictions, probs = self.make_prediction(self.X_dev)
        train_predictions, train_probs = self.make_prediction(self.X_train)

        baseline = {'recall': recall_score(self.y_dev, [1 for _ in range(len(self.y_dev))]),
                    'precision': precision_score(self.y_dev, [1 for _ in range(len(self.y_dev))]), 'roc': 0.5}

        results = {'recall': recall_score(self.y_dev, predictions),
                   'precision': precision_score(self.y_dev, predictions),
                   'roc': roc_auc_score(self.y_dev, probs)}

        train_results = {'recall': recall_score(self.y_train, train_predictions),
                         'precision': precision_score(self.y_train, train_predictions),
                         'roc': roc_auc_score(self.y_train, train_probs)}
        for metric in ['recall', 'precision', 'roc']:
            print(
                f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

        if plot:

            # Calculate false positive rates and true positive rates
            base_fpr, base_tpr, _ = roc_curve(self.y_dev, [1 for _ in range(len(self.y_dev))])
            model_fpr, model_tpr, _ = roc_curve(self.y_dev, probs)

            plt.figure(figsize=(8, 6))
            plt.rcParams['font.size'] = 16

            # Plot both curves
            plt.plot(base_fpr, base_tpr, 'b', label='baseline')
            plt.plot(model_fpr, model_tpr, 'r', label='model')
            plt.legend()
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.show()


class Random_forest(base_model):

    def __init__(self,  X_train, y_train, X_dev, y_dev, parameters):
        super().__init__(X_train, y_train, X_dev, y_dev)
        self.model_type = RandomForestClassifier(**parameters)


    # CV search for the best model
    def grid_search(self):
        """
        Searches for the best hyperparameters for the random forest model
        :return:
        """
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]  # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                       random_state=42, n_jobs=-1)  # Fit the random search model
        rf_random.fit(self.X_train, self.y_train)

        return rf_random.best_params_


class Catboost(base_model):

    def __init__(self,  X_train, y_train, X_dev, y_dev):
        super().__init__(X_train, y_train, X_dev, y_dev)
        self.model_type = CatBoostClassifier(
                custom_loss=['Accuracy'],
                random_seed=42,
                logging_level='Silent')



def save_model(model, file_path):

    pickle.dump(model, open(file_path, 'wb'))
    print(f"Saved the model at {file_path}")


def load_model(file_path):

    loaded_model = pickle.load(open(file_path, 'rb'))
    return loaded_model



