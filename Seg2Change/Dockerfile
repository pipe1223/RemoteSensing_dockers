FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY seg2change_demo.py .
COPY src ./src
COPY scripts ./scripts
COPY sample_data ./sample_data
COPY tests ./tests
COPY README.md .
COPY docs ./docs

ENTRYPOINT ["python", "/app/seg2change_demo.py"]
CMD ["smoke-test", "--output-dir", "/tmp/seg2change-smoke"]
