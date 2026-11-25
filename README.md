\# PDF Q\&A Chatbot (RAG)



A chatbot that answers questions from your own PDF and cites the pages it used. Built with Retrieval-Augmented Generation (RAG), so answers come from the document, not from the model's memory.



\## Demo

!\[Upload](screenshots/1-upload.png)

!\[Answer with sources](screenshots/2,3-answer.png)

!\[Refuses to guess](screenshots/4-not-found.png)



\## How it works

1\. \*\*Extract:\*\* read the PDF page by page (pypdf)

2\. \*\*Chunk:\*\* split the text into overlapping chunks of about 800 characters

3\. \*\*Embed:\*\* convert each chunk into a vector (sentence-transformers, all-MiniLM-L6-v2)

4\. \*\*Retrieve:\*\* find the 4 chunks closest to the question using a FAISS index

5\. \*\*Generate:\*\* send the question and chunks to the Gemini API with instructions to answer only from the context

6\. \*\*Cite:\*\* show the answer with source pages and the retrieved text



If the answer is not in the document, the bot says so instead of guessing.



\## Tech stack

Python, pypdf, sentence-transformers, FAISS, Google Gemini API, Streamlit



\## Features

\- PDF upload and indexing in the browser

\- Answers with page citations and expandable source chunks

\- Refuses to answer when the document does not contain the answer

\- Retries on busy servers (503) and falls back to other models



\## Run it yourself

```

pip install -r requirements.txt

set GEMINI\_API\_KEY=your\_key\_here

streamlit run app\_ui.py

```

Get a free key at aistudio.google.com/apikey. A command-line version is in `app.py`.



\## Limitations

\- Works only on text-based PDFs (scanned PDFs need OCR)

\- Code samples and tables in PDFs can extract with messy spacing

\- Table-of-contents pages sometimes match questions



\## Possible next steps

\- Skip table-of-contents pages

\- Support multiple PDFs

\- Add AWS Bedrock as the LLM and S3 for file storage



\## What I improved

\- Search missed some answers at first (for example "while loops"), so I combined meaning-based search with keyword matching.

\- Skipped table-of-contents and tiny junk chunks that crowded out real content.

\- Added retries and model fallback after hitting busy-server (503) errors.



Built by Shreya Battul | www.linkedin.com/in/shreya-battul

