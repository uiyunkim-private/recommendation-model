from src.main.python.feature_extractor.feature_extractor import FeatureExtractor
from src.main.python.s3.reader import S3Reader


import pandas as pd

class FeatureExtractionPipeline:
    def __init__(self, bucket_name, training_data_key, test_data_key):
        # Load data from S3
        train = S3Reader(bucket_name, training_data_key)
        test = S3Reader(bucket_name, test_data_key)

        self.training = train.execute()
        self.test = test.execute()

        self.label = []
        for i in range(100):
            self.label.append('label_'+str(i))



        self.x_training = self.training.drop(self.label, axis=1)
        print(self.x_training)
        self.test = self.test.drop(self.label, axis=1)

    def process(self, bucket_name):
        feature_extractor = FeatureExtractor(self.x_training, self.test)

        train, test = feature_extractor.transform()
        feature_extractor.save_to_s3(bucket_name, 'prediction/column_transformer.pkl')

        return train, self.training[self.label], test
