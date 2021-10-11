FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN pip install pipenv

RUN pipenv install --system --deploy

COPY ["predict_flask.py", "model1.bin", "dv.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict_flask:app"]