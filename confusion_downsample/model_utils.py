from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


class Model():

    def __init__(self, X_train, y_train, X_dev, y_dev):
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev

    def evaluate_model(self, predictions, probs, train_predictions, train_probs):
        """
        Compare machine learning model to baseline performance.
        Computes statistics and shows ROC curve.
        :param predictions:
        :param probs:
        :param train_predictions:
        :param train_probs:
        :return:
        """

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

    def train_RF(self, return_classifier=False):
        """

        :param feature_columns:
        :param target_column:
        :return:
        """
        rfc = RandomForestClassifier()
        rfc.fit(self.X_train, self.y_train)

        if return_classifier:
            return rfc
        
        else:
            train_rf_predictions = rfc.predict(self.X_train)
            train_rf_probs = rfc.predict_proba(self.X_train)[:, 1]
            # Actual class predictions
            rfc_predict = rfc.predict(self.X_dev)
            # Probabilities for each class
            rfc_probs = rfc.predict_proba(self.X_dev)[:, 1]
            self.evaluate_model(rfc_predict, rfc_probs, train_rf_predictions, train_rf_probs)
            plt.savefig('roc_auc_curve.png')