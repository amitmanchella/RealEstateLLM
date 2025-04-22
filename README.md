# RealEstateLLM

Steps to run (put these commands in your terminal) :
1. `python -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python main.py --process --force`
5. `python main.py --embeddings`
6. `python main.py --chat`

To exit chatbot:
1. `exit`

If you want to generate data for fine-tuning an LLM (after running till 5 above) :
1. `python main.py --finetune`

To deactivate venv :
1. `deactivate`

USE `python3` if `python` is not working.
