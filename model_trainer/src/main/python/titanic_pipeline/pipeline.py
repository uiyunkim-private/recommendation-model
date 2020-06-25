from src.main.python.titanic_pipeline.model_pipeline import TitanicModelPipeline


# Execute this pipeline offline to update feature extractors & model in AWS S3
def main():
    pipeline = TitanicModelPipeline(
        bucket_name='team-03-bucket',
        training_data_key='prediction/train.csv',
        test_data_key='prediction/test.csv'
    )
    pipeline.process()


if __name__ == '__main__':
    main()
