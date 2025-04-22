# RealEstateLLM

Steps to run (put these commands in your terminal):

1. `python -m venv venv`
2. `source venv/bin/activate`
3. `pip install pandas numpy faiss-cpu PyPDF2 sentence-transformers python-docx tqdm ipywidgets`
4. `python main.py --process --force`
5. `python main.py --embeddings`
6. `python main.py --chat`

If you want to generate data for fine-tuning an LLM:
1. `python main.py --finetune`

USE `python3` if `python` is not working.
