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

        mapped = {k: [v] for k, v in data[0].items()}
        df = pd.DataFrame.from_dict(mapped)
        for i in range(1,len(data)):
            mapped = {k: [v] for k, v in data[i].items()}
            df = df.append(pd.DataFrame.from_dict(mapped),ignore_index=True)

        #print(df)
        transformed = column_transformer.transform(df)
        answers = model.predict(transformed, )


        print(answers)
        users = {}
        for i,answer in enumerate(answers):
            labels = {}
            for j, data in enumerate(answer):
                labels.update({'label_'+str(j):str(data)})
            users.update({"user"+str(i):json.dumps(labels)})



        print(json.dumps(users))
        return json.dumps(users)

    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()
