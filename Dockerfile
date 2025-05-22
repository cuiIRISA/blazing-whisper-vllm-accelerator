FROM vllm/vllm-openai:latest

# Create app directory
WORKDIR /opt/ml/code

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app /opt/ml/code/app/

# Copy serve.py script (make it executable)
COPY serve /opt/ml/code/
RUN chmod +x /opt/ml/code/serve

# Set Python path to include code directory
ENV PYTHONPATH=/opt/ml/code:${PYTHONPATH}

# Configure Gunicorn workers
ENV GUNICORN_WORKERS=1

# Expose SageMaker port
EXPOSE 8080

# Start the SageMaker-compatible server
ENTRYPOINT ["python3", "/opt/ml/code/serve"]