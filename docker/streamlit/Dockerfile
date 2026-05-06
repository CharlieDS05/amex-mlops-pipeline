FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.streamlit.txt .
RUN pip install --no-cache-dir -r requirements.streamlit.txt

# Copy the demo code
COPY src/demo /app/src/demo

# Streamlit serves on port 8501 by default
EXPOSE 8501

# Run Streamlit (--server.address=0.0.0.0 makes it accessible from outside the container)
CMD ["streamlit", "run", "src/demo/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]