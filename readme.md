## Readme.md

Installer ollama :
[ollama website](https://ollama.com/)

puis depuis le cmd windows, installer le model suivant : Qwen2.5 7B Instruct.
```bash
ollama pull qwen2.5:7b-instruct
```
pull le projet
```bash
git clone https://github.com/gabi0s/rag-local.git
```
Puis mettre des fichier .txt .pdf dans le dossier data/raw

Lancer le fichier `ingest.py`
```python
python ingest.py
```

(optionel)
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