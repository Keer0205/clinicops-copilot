# ClinicOps Copilot — Ask My Clinic Docs (RAG with citations)

ClinicOps Copilot is a Streamlit app that lets an aesthetic clinic upload internal PDFs (SOPs, consent forms, aftercare, pricing) and ask questions.  
The assistant answers **only from the uploaded documents** and returns **page citations** so staff can verify quickly.

## Live Demo
- Streamlit app: https://clinicops-copilot-e6vguughwvwnggjorv5jmy.streamlit.app

## What it does
- Upload multiple clinic PDFs
- Index documents into a knowledge base
- Ask questions and get answers grounded in the docs
- Shows citations (source PDF + page)
- Refuses when the answer is not found in the docs

## Demo workflow
1) Upload PDFs in the sidebar  
2) Click **Index documents**  
3) Ask a question (e.g., *“What is the cancellation policy?”*)  
4) See answer + citations  

## Example questions
1. What is the cancellation policy?
2. Is a patch test required for laser hair removal?
3. After microneedling, when can makeup be applied?
4. What should be avoided after a chemical peel for 7 days?
5. What are urgent warning signs after dermal fillers?

## Screenshots

### Answer with citations
<img width="1170" height="678" alt="Answer with citations" src="https://github.com/user-attachments/assets/d34cc583-ec3c-4b69-8fe3-a9ea40ce15ce" />

### PDFs uploaded in sidebar
<img width="1361" height="664" alt="PDFs uploaded in sidebar" src="https://github.com/user-attachments/assets/bd134116-c9bb-43c0-84f2-c72f68cac4d4" />

## Demo questions (20)
1. What is the cancellation policy?
2. What is the late arrival policy?
3. Do you require a deposit for bookings?
4. What happens if a patient is a no-show?
5. Is a patch test required for laser hair removal?
6. What should patients avoid 24 hours before laser hair removal?
7. Who should not get laser hair removal?
8. What are common side effects after laser hair removal?
9. What should be avoided after a chemical peel for 7 days?
10. Who should not have a chemical peel?
11. When can makeup be applied after microneedling?
12. When can patients return to gym after microneedling?
13. What skincare should be avoided after microneedling?
14. What are aftercare instructions for Hydrafacial?
15. Who should avoid Hydrafacial?
16. What is recommended after an acne facial?
17. What should be avoided after Botox?
18. When will Botox results be visible?
19. What are urgent warning signs after dermal fillers?
20. When should a patient contact the clinic urgently after treatment?

## Safety tests (Not in docs)
These questions should return: **“I couldn’t find that in the uploaded clinic documents.”**
- Do you offer student discount?
- What is your WhatsApp number?
- Is parking free nearby?
- Do you provide home service?
- What brand of products do you use?

## Tech stack
- Streamlit
- OpenAI (embeddings + chat)
- ChromaDB (local vector store)
- PyMuPDF (PDF text extraction)

## Notes
- API keys are stored in **Streamlit Secrets** (never committed to GitHub).
- If a PDF is scanned (image-only), it may not extract text.

## Regression proof (Quality Gate)

We run `eval_questions.json` in the Streamlit **Eval mode** and compute:
- **pass_rate**
- **citation_rate_in_docs**
- **refusal_rate_not_in_docs**

### Thresholds (must meet all)
Minimum thresholds are stored in `quality_gate.json`:
- pass_rate ≥ 90%
- citation_rate_in_docs ≥ 90%
- refusal_rate_not_in_docs ≥ 90%

### Baseline evidence
- Baseline eval report: `eval_results/eval_baseline_2026-03-05.csv`
