FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY seg2change_demo.py .
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY README.md .
COPY docs ./docs
COPY vendor ./vendor

RUN mkdir -p /app/upstream \
    && unzip -q /app/vendor/Seg2Change-main.zip -d /app/upstream \
    && mv /app/upstream/Seg2Change-main /app/upstream/Seg2Change \
    && mkdir -p /app/upstream/Seg2Change/weights/cach /app/upstream/Seg2Change/weights/dinov2 /app/upstream/Seg2Change/weights/sam3 \
    && cp /app/upstream/Seg2Change/exp/CK/best.pth /app/upstream/Seg2Change/weights/cach/best.pth \
    && printf 'Mount dinov2_vitb14_pretrain.pth here for real GPU runs.\n' > /app/upstream/Seg2Change/weights/dinov2/README.txt \
    && printf 'Mount sam3.pt here for real GPU runs.\n' > /app/upstream/Seg2Change/weights/sam3/README.txt

ENTRYPOINT ["python", "/app/seg2change_demo.py"]
CMD ["smoke-test", "--output-dir", "/tmp/seg2change-smoke"]
