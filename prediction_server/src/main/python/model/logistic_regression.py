import tempfile
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression,Ridge,Lasso
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import SGDClassifier
from src.main.python.utils.aws import build_s3
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
class LogisticRegressionModel:
    def __init__(self, x_train, y_train):
        self.s3 = build_s3()
        #self.model = LogisticRegression(solver='lbfgs')
        self.model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=14,weights='distance'))
        #self.model = MultiOutputRegressor(Ridge())
        self.x_train = x_train
        self.y_train = y_train

    def fit(self, sample_weight=None):

        self.model.fit(self.x_train, self.y_train, sample_weight)

    def predict(self, test_x):
        #return self.model.predict(test_x)

        labels = self.model.classes_
        answers = self.model.predict_proba(test_x,)

        arr = []
        for i in range(len(labels)):
            label = labels[i]
            answer = answers[i]

            predicted = []

            if len(label) is 1 and label[0] is 1:
                predicted.append(1)
            print(answer)
            for each in answer:

                if label[np.argmax(each)] == 1:
                    #print(each[np.argmax(each)])
                    if each[np.argmax(each)] > 0.95:
                        predicted.append(label[np.argmax(each)])
                    else:

                        sliced_label = label[1:]
                        sliced_each = each[1:]
                        predicted.append(sliced_label[np.argmax(sliced_each)])
                else:
                    predicted.append(label[np.argmax(each)])

            arr.append(predicted)
        final = np.array(arr)
        final = np.flipud(np.rot90(final))
        print(final)
        return final

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
