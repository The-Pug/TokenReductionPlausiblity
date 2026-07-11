#!/bin/sh
ollama serve >/tmp/ollama.log 2>&1 &

i=0
until ollama list >/dev/null 2>&1; do
    i=$((i+1))
    if [ "$i" -ge 60 ]; then
        echo "ollama not ready after 30s — continuing (remote-only mode)" >&2
        break
    fi
    sleep 0.5
done

exec python3 -m src.agent.main
