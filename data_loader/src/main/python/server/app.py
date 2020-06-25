import pandas as pd
from flask import Flask, request, jsonify
import json

from feature_extractor.feature_extractor import FeatureExtractor
from model.logistic_regression import LogisticRegressionModel
from utils.aws import build_s3


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
        prediction = model.predict(transformed)
	
        users = {}

        for i,data in enumerate(prediction):
            predictions = {}
            for j ,answer in enumerate(data):
                predictions.update({'label_'+str(j):answer})
            users.update({'user_'+str(i):json.dumps(predictions)})


        return json.dumps(users)

    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()
