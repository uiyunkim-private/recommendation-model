
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 531471861132.dkr.ecr.ap-northeast-2.amazonaws.com/



docker build -t 531471861132.dkr.ecr.ap-northeast-2.amazonaws.com/team3-prediction-server .

docker push 531471861132.dkr.ecr.ap-northeast-2.amazonaws.com/team3-prediction-server