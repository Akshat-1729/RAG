import feedparser
from llm import ask_llm
from sentence_transformers import SentenceTransformer   
import faiss
import numpy as np
from bs4 import BeautifulSoup
feeds = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://www.theguardian.com/world/rss",
]


data = []

def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

cleaned_data = []

def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


for url in feeds:
    feed = feedparser.parse(url)
    for entry in feed.entries[:50]:
        title = entry.title.strip()
        published = entry.published.strip() if "published" in entry else ""
        summary_raw = entry.summary.strip() if "summary" in entry else ""
        summary = clean_html(summary_raw)
        text = f"""News Article
        Title: {title}
        Published: {published}
        Summary: {summary}
        Topic: World News"""

        data.append(text.strip())

chunked_data = []

for item in data:
    chunks = chunk_text(item)
    chunked_data.extend(chunks)



embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = embedding_model.encode(chunked_data)
print(embeddings.shape)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
query = "Why is there tension in the Middle East?"

query_embedding = embedding_model.encode([query])

k = 3
distances, indices = index.search(query_embedding, k)

results = [chunked_data[i] for i in indices[0]]

context = "\n\n".join(results)

query = "Why is there tension in the Middle East?"

answer = ask_llm(context, query)

print(answer)