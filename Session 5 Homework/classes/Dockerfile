FROM python:3.6.15-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY ["predict_flask.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict_flask:app"]