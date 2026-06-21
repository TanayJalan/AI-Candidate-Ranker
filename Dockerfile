FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt $HOME/app/
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . $HOME/app/
RUN mkdir -p data/output data/processed data/raw

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.headless=true"]
