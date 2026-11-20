import chromadb, numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

client = chromadb.PersistentClient(path="./db")
col    = client.get_collection("social_e5_large")   # ← your collection

print("docs in collection:", col.count())

d = col.get(include=["embeddings", "documents", "metadatas"])
V = np.array(d["embeddings"], dtype=np.float32)
print("matrix shape:", V.shape, "  dtype:", V.dtype)   # (783, 1024) float32
print("norm example:", np.linalg.norm(V[0]))           # ≈ 1.0 if normalized
print("first 8 dims :", V[0][:8])

# per-dimension statistics — a quick "what does my space look like" view
print("dim mean/std/min/max:",
      V.mean(0).mean(), V.std(0).mean(), V.min(), V.max())

# 2-D projection (drop the scatter part if headless)
coords = TSNE(n_components=2, perplexity=30, init="pca",
              random_state=0).fit_transform(V)
print("projection shape:", coords.shape)
