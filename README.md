# ClinicOps Copilot — Ask My Clinic Docs

ClinicOps Copilot is a document-based AI assistant built for clinics that need quick, reliable answers from their internal documents.

Instead of manually searching through SOPs, consent forms, aftercare guides, and pricing PDFs, staff can upload those files, ask a question in plain English, and get a grounded answer with page-level citations.

The goal of this project is simple: **help clinic teams find the right information faster, with less guesswork and more confidence**.

---

## Why I Built This

In many clinics, important information is spread across multiple PDFs. Staff may need to check cancellation rules, aftercare instructions, patch test requirements, pricing details, or treatment warnings while supporting patients or handling daily operations.

That usually means opening several files, scanning through pages, and trying to find the correct answer under time pressure.

This process is:
- slow
- repetitive
- difficult to scale
- vulnerable to mistakes when information is missed or misread

I built ClinicOps Copilot to reduce that manual effort and show how a practical RAG application can be used in a real clinic workflow.

Rather than generating unsupported responses, the assistant answers **only from the uploaded documents** and shows **page-level citations** so the user can verify the source.

---

## What the Project Does

ClinicOps Copilot allows a clinic team to:

- upload internal clinic PDFs
- index them into a searchable knowledge base
- ask natural-language questions
- receive grounded answers with citations
- refuse to answer when the information is not found in the documents
- run evaluation checks using predefined questions and quality thresholds

This makes it a useful example of a **retrieval-augmented generation (RAG)** application focused on reliability rather than generic chatbot behavior.

---

## Example Use Cases

A clinic team could use this application to quickly answer questions like:

- What is the cancellation policy?
- Do we require a deposit for bookings?
- What is the late arrival policy?
- Is a patch test required before laser treatment?
- When can makeup be applied after microneedling?
- What should be avoided after a chemical peel?
- What are the warning signs after dermal fillers?
- When should a patient contact the clinic urgently?

These are the kinds of questions that often come up during daily operations, and they are exactly the type of questions that should be answered from approved clinic documents rather than memory or guesswork.

---

## How It Works

The workflow is designed to be simple:

1. Upload one or more clinic PDFs in the Streamlit interface  
2. Click **Index Documents**  
3. Ask a question in natural language  
4. The system retrieves the most relevant passages  
5. The assistant generates a response grounded in those passages  
6. The answer is returned with source and page-level citations  

If the answer is not supported by the uploaded files, the assistant refuses instead of inventing a response.

---

## Live Demo

- **Streamlit App:** https://clinicops-copilot-e6vguughwvwnggjorv5jmy.streamlit.app

---

## Key Features

- Upload multiple clinic PDFs
- Search across internal clinic knowledge in one place
- Ask questions in plain English
- Return answers with source PDF and page citations
- Reduce unsupported or hallucinatory responses
- Refuse when the answer is not present in the documents
- Run evaluation checks using predefined question sets
- Demonstrate a practical healthcare-focused RAG workflow

---

## Screenshots

### Answer with citations
<img width="1170" height="678" alt="Answer with citations" src="https://github.com/user-attachments/assets/d34cc583-ec3c-4b69-8fe3-a9ea40ce15ce" />

### PDFs uploaded in sidebar
<img width="1361" height="664" alt="PDFs uploaded in sidebar" src="https://github.com/user-attachments/assets/bd134116-c9bb-43c0-84f2-c72f68cac4d4" />

---

## Tech Stack

- **Python**
- **Streamlit**
- **OpenAI API** for embeddings and answer generation
- **ChromaDB** for vector storage and retrieval
- **PyMuPDF** for PDF text extraction

---

## Architecture / Workflow

- PDF upload through the Streamlit UI
- Text extraction from uploaded documents
- Chunking of extracted text
- Embedding generation for document chunks
- Storage in ChromaDB for semantic retrieval
- Retrieval of relevant passages based on the user’s question
- LLM-generated answer grounded in retrieved context
- Citation display using PDF source and page number
- Evaluation mode for regression-style testing

---

## Reliability Approach

One of the main goals of this project was to make the assistant more trustworthy than a generic chatbot.

To support that, the app is designed to:

- answer only from uploaded clinic documents
- provide page-level citations for traceability
- avoid unsupported answers
- refuse when the required information is missing
- support evaluation checks using sample question sets

This makes the project more than just a demo chat interface. It is a practical attempt to build a safer and more verifiable document Q&A workflow.

---

## Evaluation

The repository includes evaluation assets such as:

- predefined question sets
- quality thresholds
- evaluation outputs

These are used to check whether the assistant is:
- grounded in the uploaded documents
- returning citations correctly
- refusing when information is unavailable

This helps demonstrate a more thoughtful and reliability-focused RAG implementation.

---

## Folder Structure

```bash
clinicops-copilot/
│
├── app.py
├── requirements.txt
├── eval_questions.json
├── quality_gate.json
├── README.md
│
├── sample_docs/
├── eval_results/
├── scripts/
├── .github/
└── .devcontainer/
