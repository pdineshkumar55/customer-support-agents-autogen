# ShopNest AI Support Desk — Streamlit + AutoGen
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (this layer is cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the app code and the policy PDFs
COPY . .

EXPOSE 8501

# Lets Docker know if the app has stopped responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]