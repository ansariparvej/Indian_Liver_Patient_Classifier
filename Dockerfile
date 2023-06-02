FROM python:3.8.16-slim-buster
COPY . /app/
WORKDIR /app/
RUN pip install -r requirements.txt
RUN apt update -y && apt install awscli -y
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]