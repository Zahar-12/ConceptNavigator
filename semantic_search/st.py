import json
import pandas as pd
import numpy as np

with open("code_corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

with open("eval_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

df = pd.DataFrame(corpus)
df.head()

from sentence_transformers import SentenceTransformer

model_1 = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
model_2 = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

codes = df["code"].tolist()

embeddings_1 = model_1.encode(codes, show_progress_bar=True)
embeddings_2 = model_2.encode(codes, show_progress_bar=True)

from sklearn.metrics.pairwise import cosine_similarity

def search(query, model, corpus_embeddings, top_k=3):
    q_emb = model.encode([query])
    scores = cosine_similarity(q_emb, corpus_embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx

def precision_at_3(model, corpus_embeddings):
    correct = 0

    for q in questions:
        query = q["question"]
        true_id = q["correct_chunk_id"]

        top3 = search(query, model, corpus_embeddings, top_k=3)
        top3_ids = [corpus[i]["id"] for i in top3]

        if true_id in top3_ids:
            correct += 1

    return correct / len(questions)

p1 = precision_at_3(model_1, embeddings_1)
p2 = precision_at_3(model_2, embeddings_2)

results = pd.DataFrame({
    "model": [
        "MiniLM-L12-v2",
        "mpnet-base-v2"
    ],
    "precision@3": [p1, p2]
})

results

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

best_embeddings = embeddings_2  # обычно mpnet лучше
categories = df["category"].tolist()

tsne = TSNE(n_components=2, random_state=42)
coords = tsne.fit_transform(best_embeddings)

plt.figure(figsize=(8,6))
plt.scatter(coords[:,0], coords[:,1])

for i, cat in enumerate(categories):
    plt.text(coords[i,0], coords[i,1], cat, fontsize=6)

plt.title("t-SNE visualization")
plt.show()