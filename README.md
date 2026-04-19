# Retrieval-Augmented Generation (RAG) News Assistant

This project is a complete, end-to-end implementation of a Retrieval-Augmented Generation (RAG) system. It automatically fetches the latest news from RSS feeds, indexes the data using vector embeddings, and uses a Large Language Model (LLM) to answer user questions based specifically on the ingested news context.

## What is RAG?
Retrieval-Augmented Generation is a technique that bridges the gap between a user's prompt and an LLM's static knowledge base. Instead of relying solely on what the LLM was trained on, RAG *retrieves* relevant, up-to-date information from a custom database and *augments* the prompt with this context, asking the LLM to generate an answer based on it.

## Architecture & Data Flow

The system operates in a linear pipeline, transforming raw web data into a precise, AI-generated answer. Here is the step-by-step flow:

### 1. Data Ingestion (Data Calling)
The pipeline starts by fetching live data. We use the `feedparser` library to pull the latest RSS feeds from major news outlets (BBC and The Guardian). For each news entry, we extract the title, publication date, and summary.

### 2. Preprocessing & Cleaning
Raw RSS data often contains HTML tags. We use `BeautifulSoup` to strip out HTML tags and format the text cleanly. 

### 3. Chunking
LLMs have context window limits, and vector searches are less accurate on massive blocks of text. To fix this, we split the cleaned news articles into smaller, manageable "chunks" of text (approx. 100 words each). This ensures that our eventual semantic search retrieves highly specific paragraphs rather than entire articles.

### 4. Vectorization & Embeddings
This is the core of the RAG retrieval mechanism.
- We use the `sentence-transformers` library (specifically the `all-MiniLM-L6-v2` model) to convert every text chunk into a **Vector Embedding**. 
- An embedding is a high-dimensional mathematical representation of the text's *meaning*. If two chunks of text have a similar meaning, their vectors will be close together in this multi-dimensional space.

### 5. Vector Database (FAISS)
Once we have our vectors, we need a way to search them quickly. We use **FAISS (Facebook AI Similarity Search)**. FAISS builds an index (`IndexFlatL2`) that stores our embeddings and allows us to perform lightning-fast similarity searches using L2 (Euclidean) distance.

### 6. Querying & Retrieval
When a user asks a question (e.g., *"Why is there tension in the Middle East?"*):
1. The question is passed through the exact same embedding model to create a **Query Vector**.
2. We ask FAISS to find the top $k$ (e.g., $k=3$) vectors in our database that are mathematically closest to the query vector.
3. We map those closest vectors back to their original text chunks. These chunks represent the most relevant news information to the user's question.

### 7. LLM Generation
Finally, we take the retrieved text chunks and combine them into a single `context` string. We pass this context, along with the user's original query, to an LLM via the OpenRouter API. The LLM is instructed to act as a helpful assistant and answer the user's question *only* using the provided context.

## Technologies Used
- **Data Collection**: `feedparser`, `beautifulsoup4`
- **Embeddings**: `sentence-transformers` (Hugging Face)
- **Vector Search**: `faiss-cpu`
- **LLM API**: OpenRouter (`requests`), `python-dotenv` for environment management

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install feedparser sentence-transformers faiss-cpu numpy beautifulsoup4 requests python-dotenv
   ```
2. **Environment Variables**:
   Ensure you have a `.env` file in the root directory with your OpenRouter API key:
   ```
   API_KEY="your_openrouter_api_key_here"
   ```
3. **Run the script**:
   ```bash
   python news_data.py
   ```
