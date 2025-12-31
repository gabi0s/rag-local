## Readme.md

pour utiliser ollama sur mon gpu AMD RX5600XT
```shell
set OLLAMA_VULKAN=1
ollama serve
```

Poser une question dans le RAG
```python
python scripts\ask.py "ta question"
```

insaller le model ollama
```shell
ollama pull qwen2.5:7b-instruct
```