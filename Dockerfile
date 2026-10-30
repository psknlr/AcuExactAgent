FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md requirements.txt ./
COPY acuexact ./acuexact
RUN pip install --no-cache-dir .
RUN useradd --create-home appuser
USER appuser
EXPOSE 8765
# Publish to the host's loopback: docker run -p 127.0.0.1:8765:8765 acuexact
CMD ["python", "-m", "acuexact", "serve", "--host", "0.0.0.0", "--port", "8765"]
