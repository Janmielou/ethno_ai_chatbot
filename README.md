# Arcana Herbarium – Mindanao Ethnomedicinal Plant Chatbot

A local, console-based chatbot that answers questions about ethnomedicinal plants recorded in Mindanao, Philippines.  
The chatbot only uses the **CSV dataset** as its source of truth and refuses to guess outside the dataset.

> ⚠️ **Disclaimer:**  
> This tool is for **research and educational purposes only**.  
> It is **not** medical advice and must not be used to diagnose, treat, or cure any disease.

---

## 1. Features

- Answers questions about ethnomedicinal plants in the dataset:
  - local name, scientific name, family  
  - diseases/conditions used on  
  - parts used, preparation & administration  
  - dosage and administration frequency  
  - reported side effects (if any)
- Condition-specific questions, for example:  
  - “Can Tawa-tawa be used to treat dengue?”  
  - “Is Lagundi used for cough?”
- Preparation-method questions, for example:  
  - “What is the preparation method for Lagundi for treating cough?”  
- Multi-plant search:
  - “List all plants that can help with cough”
  - “Plants that use leaves as medicinal parts”
- Dataset-only answers (RAG with **Chroma** + **Ollama** embeddings and LLaMA 3.2).

---

## 2. Project Structure

```text
ai_chatbot/
├─ main.py        # Chat loop, question routing, printing with Rich
├─ vector.py      # CSV loading, Chroma vector DB, and dataset helpers
├─ datasets/
│  └─ Mindanao_Ethnomedicinal_Plant_Dataset.csv
└─ chroma_mindanao_ethno_db/  # Auto-created vector store directory
