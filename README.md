# 🏡 RealEstateLLM

A lightweight local chatbot and toolset for analyzing real estate documents, generating embeddings, and supporting fine-tuning of LLMs.

---

## 🔧 Setup & Run

Open your terminal and run the following commands:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Process real estate documents
python main.py --process --force

# 5. Generate sentence embeddings
python main.py --embeddings

# 6. Launch the interactive chatbot
python main.py --chat
```

> ⚠️ Use `python3` instead of `python` if needed.

---

## 💬 Exit the Chatbot

```bash
exit
```

---

## 📊 Generate Data for LLM Fine-tuning

If you'd like to prepare data for fine-tuning an LLM (after completing steps 1–5), run:

```bash
python main.py --finetune
```

---

## 📴 Deactivate the Virtual Environment

To deactivate the virtual environment at any time:

```bash
deactivate
```
