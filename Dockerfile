FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages openai

RUN sh -c 'ollama serve >/tmp/pull.log 2>&1 & \
    i=0; until ollama list >/dev/null 2>&1; do \
        i=$((i+1)); [ $i -ge 60 ] && cat /tmp/pull.log && exit 1; sleep 1; done; \
    ollama pull gemma3:1b'

ENV OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_NUM_PARALLEL=2 \
    LOCAL_MODEL=gemma3:1b \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY src /app/src
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
