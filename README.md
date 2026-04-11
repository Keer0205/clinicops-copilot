# ClinicOps Copilot — Ask My Clinic Docs

ClinicOps Copilot is a **RAG-based clinic document assistant** built with **Python, Streamlit, OpenAI, and ChromaDB**.

It helps clinic staff quickly find reliable answers from internal documents such as **SOPs, consent forms, aftercare guides, pricing sheets, and clinic policies**. Instead of manually searching across multiple PDFs, users can upload documents, ask a question in plain English, and receive a **grounded answer with page-level citations**.

The assistant is designed to answer **only from the uploaded documents**. If the information is not present, it **refuses instead of guessing**, making it a safer and more practical workflow tool than a generic chatbot.

---

## Business Problem

In many clinics, important operational information is spread across multiple PDFs. Staff may need to check cancellation rules, late arrival policies, patch test requirements, aftercare instructions, pricing notes, or treatment warnings while handling patients and daily operations.

This often leads to:

- slow manual searching
- repeated effort across staff
- inconsistent answers
- higher risk of missing important details

ClinicOps Copilot was built to show how a **practical, reliability-focused RAG application** can improve speed, consistency, and confidence in clinic operations.

---

## What the Project Does

ClinicOps Copilot allows users to:

- upload one or more clinic PDFs
- index them into a searchable vector database
- ask natural-language questions
- retrieve grounded answers with citations
- refuse when the answer is not found in the documents
- run evaluation checks using predefined question sets

---

## Example Questions

- What is the cancellation policy?
- Do we require a deposit for bookings?
- What is the late arrival policy?
- Is a patch test required before laser treatment?
- When can makeup be applied after microneedling?
- What should be avoided after a chemical peel?
- What are the urgent warning signs after dermal fillers?

---

## Live Demo

- **Streamlit App:** https://clinicops-copilot-e6vguughwvwnggjorv5jmy.streamlit.app

---

## Tech Stack

- **Python**
- **Streamlit**
- **OpenAI API**
- **ChromaDB**
- **PyMuPDF**

---

<img width="1365" height="613" alt="image" src="https://github.com/user-attachments/assets/cd0d5742-a185-4129-9061-ac81f1f0d228" />

<img width="1365" height="594" alt="image" src="https://github.com/user-attachments/assets/093b9e44-dc7b-447c-b5c5-e75c011b487c" />


## Key Features

- Upload multiple clinic PDFs
- Ask questions in plain English
- Retrieve answers grounded in uploaded documents
- Return source and page-level citations
- Refuse out-of-document questions
- Run evaluation checks for citation/refusal behaviour

---

## How It Works

1. Upload one or more clinic PDFs  
2. Click **Index Documents**  
3. Ask a question in natural language  
4. Retrieve the most relevant passages  
5. Generate an answer grounded in retrieved context  
6. Return the answer with source and page citations  

If the answer is not supported by the uploaded files, the assistant refuses instead of inventing a response.
