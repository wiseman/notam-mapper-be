# Use an official Python runtime as the base image
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt notamai.py server.py ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "server:app"]
CMD ["python", "server.py"]
