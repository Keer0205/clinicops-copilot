# ClinicOps Copilot — Ask My Clinic Docs (RAG with Citations)

ClinicOps Copilot is a **Python + Streamlit** application that allows aesthetic clinics to upload internal PDFs such as SOPs, consent forms, aftercare guides, and pricing documents, then ask questions and receive **citation-backed answers**.

The assistant answers **only from the uploaded documents** and provides **page-level citations**, helping staff verify information quickly and reducing the risk of unsupported answers.

## Business Problem

Clinic staff often need to search across multiple SOPs, consent forms, aftercare guides, and pricing PDFs to answer operational and patient-support questions. This process can be slow, inconsistent, and error-prone.

ClinicOps Copilot was built to reduce manual searching and provide faster, grounded answers based only on approved clinic documents.

## Solution Overview

The application ingests uploaded clinic PDFs, extracts and chunks the text, indexes the content into a searchable vector store, and retrieves relevant passages to generate grounded answers with citations.

If the answer is not present in the source documents, the assistant refuses rather than guessing.

## Live Demo

- **Streamlit app:** https://clinicops-copilot-e6vguughwvwnggjorv5jmy.streamlit.app

## Key Features

- Upload multiple clinic PDFs
- Index documents into a searchable knowledge base
- Ask natural-language questions about uploaded documents
- Return answers with source PDF and page-level citations
- Refuse when the answer is not supported by the documents
- Run evaluation checks using predefined question sets and thresholds

## Demo Workflow

1. Upload PDFs in the sidebar  
2. Click **Index Documents**  
3. Ask a question such as **“What is the cancellation policy?”**  
4. Review the answer and supporting page citations  

## Example Questions

1. What is the cancellation policy?
2. What is the late arrival policy?
3. Do you require a deposit for bookings?
4. Is a patch test required for laser hair removal?
5. After microneedling, when can makeup be applied?
6. What should be avoided after a chemical peel for 7 days?
7. Who should avoid chemical peels?
8. What are the aftercare instructions for dermal fillers?
9. What are urgent warning signs after dermal fillers?
10. When should a patient contact the clinic urgently after treatment?

## Screenshots

### Answer with citations
<img width="1170" height="678" alt="Answer with citations" src="https://github.com/user-attachments/assets/d34cc583-ec3c-4b69-8fe3-a9ea40ce15ce" />

### PDFs uploaded in sidebar
<img width="1361" height="664" alt="PDFs uploaded in sidebar" src="https://github.com/user-attachments/assets/bd134116-c9bb-43c0-84f2-c72f68cac4d4" />

## Tech Stack

- Python
- Streamlit
- OpenAI API (embeddings + chat)
- ChromaDB (local vector store)
- PyMuPDF (PDF text extraction)

## Architecture / Workflow

- PDF upload through the Streamlit UI
- Text extraction using PyMuPDF
- Text chunking and embedding generation
- ChromaDB vector storage for retrieval
- LLM-based answer generation grounded in retrieved passages
- Citation display using source PDF and page number
- Evaluation mode for regression-style testing

## How to Run

```bash
git clone https://github.com/Keer0205/clinicops-copilot.git
cd clinicops-copilot
pip install -r requirements.txt
streamlit run app.py
