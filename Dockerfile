FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY image_utils.py .
COPY ml_model.py .

# Create uploads directory
RUN mkdir -p uploads

# Expose the port
EXPOSE 9000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"] 