import pandas as pd
from flask import Flask, request, jsonify
import json
import numpy as np
from src.main.python.feature_extractor.feature_extractor import FeatureExtractor
from src.main.python.model.logistic_regression import LogisticRegressionModel
from src.main.python.utils.aws import build_s3


def main():
    """
    This assumes that the offline model pipeline was already processed,
    meaning the feature extractors & model can be loaded from AWS S3
    """

    app = Flask(__name__)

    # build AWS S3 Client & point to the bucket
    s3 = build_s3()
    bucket_name = 'team-03-bucket'

    # point to the location storing feature extractor & offline trained model
    column_transformer = FeatureExtractor.load_from_s3(s3, bucket_name, 'prediction/column_transformer.pkl')
    model = LogisticRegressionModel.load_from_s3(s3, bucket_name, 'prediction/model.pkl')

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        mapped = {k: [v] for k, v in data.items()}
        df = pd.DataFrame.from_dict(mapped)
        transformed = column_transformer.transform(df)

        labels = model.classes_
        answers = model.predict_proba(transformed, )

        arr = []
        for i in range(len(labels)):
            label = labels[i]
            answer = answers[i]

            predicted = []

            if len(label) is 1 and label[0] is 1:
                predicted.append(1)

            for each in answer:

                if label[np.argmax(each)] == 1:
                    # print(each[np.argmax(each)])
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

        each = []
        for user in final:
            toget = []
            for data in user:
                toget.append(int(data))
            each.append(toget)

        users = {}
        for i,data in enumerate(each):
            predictions = {}
            for j ,answer in enumerate(data):
                predictions.update({'label_'+str(j):answer})
            users.update({'user_'+str(i):json.dumps(predictions)})

        return json.dumps(users)

    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()
