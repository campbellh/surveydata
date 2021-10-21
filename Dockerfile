FROM python:3.9

EXPOSE 8080

ADD my_app.py .

ADD logo.png .

ADD data.csv .

RUN pip install streamlit pandas matplotlib seaborn pywaffle pillow

#CMD ["python", "my_app.py"]

CMD streamlit run --server.port 8080 --server.enableCORS false my_app.py