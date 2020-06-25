import requests
import csv
import json
import random
import os
import time
import numpy as np
from src.main.python.utils.aws import build_s3


# Execute this pipeline offline to update feature extractors & model in AWS S3
def main():
    basic_id_url = "https://nqnjwccsg0.execute-api.ap-northeast-2.amazonaws.com/beta_0510/user/basic/id"
    basic_info_url = "https://nqnjwccsg0.execute-api.ap-northeast-2.amazonaws.com/beta_0510/user/basic/info"

    data = requests.get(url=basic_info_url, data=json.dumps({'body': {'mode': 0}}))
    data = data.json()

    all_user = data['body']

    random.shuffle(all_user)
    dataset_train = []
    count = 0
    for user in all_user:
        new_dict = user

        if new_dict['gender'] == 0:
            del new_dict['gender']
            new_dict.update({'gender': 'female'})
        else:
            del new_dict['gender']
            new_dict.update({'gender': 'male'})

        keys = list(new_dict.keys())

        keys.remove('id')
        keys.remove('height')
        keys.remove('weight')
        keys.remove('gender')
        keys.remove('age')
        no_rate_count = 0
        rate_count = 0
        for i, key in enumerate(keys):
            if new_dict[key] is None:
                new_dict[key] = 1
                no_rate_count = no_rate_count + 1
            else:
                rate_count = rate_count + 1
        if no_rate_count < 80:
            for i, key in enumerate(keys):
                unit = {'height': new_dict['height'], 'weight': new_dict['weight'], 'gender': new_dict['gender'],
                        'age': new_dict['age']}
                # unit = {'gender': new_dict['gender']}
                for j, k in enumerate(keys):
                    # if(k == key):
                    # pass
                    ###unit.update({'rating_' + str(j): 1})
                    # unit.update({'rating_' + str(j): new_dict[k]})
                    # else:
                    unit.update({'rating_' + str(j): new_dict[k]})
                ###unit.update({'label':new_dict[key]})

                fake_keys = list(unit.keys())
                fake_keys.remove('height')
                fake_keys.remove('weight')
                fake_keys.remove('gender')
                fake_keys.remove('age')
                # fake_keys.remove('label')

                for fake_key in fake_keys.copy():
                    if (unit[fake_key] == 1):
                        fake_keys.remove(fake_key)

                for j in range(5):
                    random.shuffle(fake_keys)
                    key_to_drop = fake_keys[0]
                    fake_keys.remove(key_to_drop)

                    after = unit.copy()
                    unit[key_to_drop] = 1
                    before = unit.copy()

                    keys_after = list(after.keys())
                    keys_after.remove('height')
                    keys_after.remove('weight')
                    keys_after.remove('gender')
                    keys_after.remove('age')

                    for l, key_after in enumerate(keys_after):
                        before.update({"label_" + str(l): after[key_after]})

                    print(before.values())
                    dataset_train.append(before)

    random.shuffle(dataset_train)
    train_group = dataset_train[:int(len(dataset_train) * 0.99)]
    test_group = dataset_train[int(len(dataset_train) * 0.99):]

    random.shuffle(train_group)
    print(train_group[0])
    print(len(train_group))
    keys = train_group[0].keys()
    path = os.path.join(os.getcwd(), 'train.csv')
    f = open(path, "w")
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(train_group)
    f.close()

    keys = test_group[0].keys()
    path = os.path.join(os.getcwd(), 'test.csv')
    f = open(path, "w")
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(test_group)
    f.close()

    s3 = build_s3()

    bucket_name = 'team-03-bucket'
    training_data_key = 'prediction/train.csv'
    test_data_key = 'prediction/test.csv'

    with open('train.csv', 'r') as fp:
        fp.seek(0)
        s3.Bucket(bucket_name).put_object(Body=fp.read(), Bucket=bucket_name, Key=training_data_key)
        fp.close()

    with open('test.csv', 'r') as fp:
        fp.seek(0)
        s3.Bucket(bucket_name).put_object(Body=fp.read(), Bucket=bucket_name, Key=test_data_key)
        fp.close()


if __name__ == '__main__':
    main()

