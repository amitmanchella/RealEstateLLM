# RealEstateLLM

Steps to run (put these commands in your terminal):
-2. python -m venv venv
-1. source venv/bin/activate
0. pip install pandas numpy faiss-cpu PyPDF2 sentence-transformers python-docx tqdm ipywidgets
1. python main.py --process --force
2. python main.py --embeddings
3. python main.py --chat

If you want to generate data for fine-tuning an LLM:
   python main.py --finetune

USE `python3` if `python` is not working.
