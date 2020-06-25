import tempfile
import numpy as np
import json
from joblib import dump, load
from sklearn.linear_model import LogisticRegression,Ridge,Lasso
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier, ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC,SVC , SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from src.main.python.utils.aws import build_s3
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
class LogisticRegressionModel:
    def __init__(self, x_train, y_train):
        self.s3 = build_s3()
        #self.model = MultiOutputClassifier(LogisticRegression(solver='lbfgs',max_iter=999999,verbose=2))
        self.model = MultiOutputClassifier(LogisticRegression(solver='lbfgs',max_iter=99999,verbose=2,n_jobs=-1))
        #self.model = OneVsRestClassifier(LogisticRegression(solver='lbfgs',max_iter=99999,verbose=2))
        self.x_train = x_train


        self.y_train = y_train

    def fit(self, sample_weight=None):

        self.model.fit(self.x_train, self.y_train)

    def predict(self, test_x):
        answers = self.model.predict(test_x,)

        return answers

    def score(self):
        return self.model.score(self.x_train, self.y_train)

    def save_to_s3(self, bucket_name, key):
        with tempfile.TemporaryFile() as fp:
            dump(self.model, fp)
            fp.seek(0)
            self.s3.Bucket(bucket_name).put_object(Body=fp.read(), Bucket=bucket_name, Key=key)
            fp.close()

    @staticmethod
    def load_from_s3(s3, bucket_name, key):
        with tempfile.TemporaryFile() as fp:
            s3.Bucket(bucket_name).download_fileobj(Fileobj=fp, Key=key)
            fp.seek(0)
            model = load(fp)
            fp.close()
        return model

    @staticmethod
    def save_predictions_to_s3(df, bucket_name, key):
        df.to_csv(f's3://{bucket_name}/{key}', index=False)
