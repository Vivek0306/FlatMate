# FlatMate

## Overview
This is a Retrieval-Augmented Generation (RAG) based command-line application that answers queries using a pre-built vector database of property-related documents.
It retrieves relevant document chunks and generates responses grounded strictly in the retrieved context.

## How it works
1. Documents are embedded and stored in a persistent vector database.
2. When a question is asked:
    * Relevant document chunks are retrieved using semantic search.
    * The retrieved context is passed to a language model.
    * The model generates a response based only on that context.
3. A lightweight CLI spinner is shown while the response is generated.


## Running the Application

Follow the steps below to run the application locally.

### 1. Create and activate a virtual environment

> Recommended as the dependecies are of large file size (may not want to mess up installing globally).

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python3 app.py
```

