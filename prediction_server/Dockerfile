FROM python:3.7-buster

ARG AWS_ACCESS_KEY_ID=AKIAXXPRRAWGIVPKHXSN
ENV AWS_ACCESS_KEY_ID ${AWS_ACCESS_KEY_ID}
ARG AWS_SECRET_ACCESS_KEY=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
ENV AWS_SECRET_ACCESS_KEY ${AWS_SECRET_ACCESS_KEY}
ARG AWS_DEFAULT_REGION=ap-northeast-2
ENV AWS_DEFAULT_REGION ${AWS_DEFAULT_REGION}
ENV PYTHONPATH "${PYTHONPATH}:/app/src/main/python"

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python" ]

EXPOSE 5000

CMD [ "src/main/python/server/app.py" ]
