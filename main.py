import pandas as pd
import numpy as np
import pyodbc
import requests
import os
import time
from cleaner import clean_resume

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not found. Set it as an environment variable.")

EMBEDDING_URL = "https://models.inference.ai.azure.com/embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"

# --------------------------------------------------
# EMBEDDING FUNCTION (DIRECT REST CALL)
# --------------------------------------------------
def get_embedding(text):
    if text is None or str(text).strip() == "":
        return None

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": str(text)
    }

    response = requests.post(EMBEDDING_URL, headers=headers, json=payload)

    if response.status_code == 429:
        time.sleep(20)  # wait before retry
        response = requests.post(EMBEDDING_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print("❌ Embedding error:", response.text)
        return None

    time.sleep(4)  # prevent rate limit
    return response.json()["data"][0]["embedding"]

# --------------------------------------------------
# LOAD CSV DATA
# --------------------------------------------------
resumes = pd.read_csv("resumes.csv")
jobs = pd.read_csv("jobs.csv")

if "text" not in resumes.columns:
    raise ValueError("resumes.csv must contain a 'text' column")

if "description" not in jobs.columns:
    raise ValueError("jobs.csv must contain a 'description' column")

resumes = resumes.dropna(subset=["text"])
jobs = jobs.dropna(subset=["description"])

# --------------------------------------------------
# BIAS REMOVAL (RESPONSIBLE AI)
# --------------------------------------------------
resumes["text"] = resumes["text"].astype(str).apply(clean_resume)
jobs["description"] = jobs["description"].astype(str)

# --------------------------------------------------
# GENERATE EMBEDDINGS
# --------------------------------------------------
resume_vectors = []
resume_ids = []

for i, row in resumes.iterrows():
    emb = get_embedding(row["text"])
    if emb is not None:
        resume_vectors.append(emb)
        resume_ids.append(row["id"])

job_vectors = []
job_ids = []

for i, row in jobs.iterrows():
    emb = get_embedding(row["description"])
    if emb is not None:
        job_vectors.append(emb)
        job_ids.append(row["id"])

resume_vectors = np.array(resume_vectors)
job_vectors = np.array(job_vectors)

# --------------------------------------------------
# COSINE SIMILARITY
# --------------------------------------------------
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

scores = np.zeros((len(resume_vectors), len(job_vectors)))

for i in range(len(resume_vectors)):
    for j in range(len(job_vectors)):
        scores[i][j] = cosine_similarity(resume_vectors[i], job_vectors[j])

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------
print("\n=== FAIR RESUME–JOB MATCHING RESULTS ===\n")

for j, job_id in enumerate(job_ids):
    job_desc = jobs[jobs["id"] == job_id]["description"].values[0]
    print(f"Job {job_id}: {job_desc}")

    ranked = []
    for i, resume_id in enumerate(resume_ids):
        ranked.append((resume_id, scores[i][j]))

    ranked.sort(key=lambda x: x[1], reverse=True)

    for rank, (rid, score) in enumerate(ranked[:2], start=1):
        resume_text = resumes[resumes["id"] == rid]["text"].values[0]
        print(f"   Rank {rank}: {resume_text}")
        print(f"      Match Score: {score:.3f}")

    print("-" * 50)

# --------------------------------------------------
# SAVE TO SQL SERVER
# --------------------------------------------------
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=localhost\\SQLEXPRESS;"
    "Database=ResumeMatchingDB;"
    "Trusted_Connection=yes;"
)

cursor = conn.cursor()
cursor.execute("DELETE FROM Matches")
conn.commit()

for i, resume_id in enumerate(resume_ids):
    for j, job_id in enumerate(job_ids):
        cursor.execute(
            "INSERT INTO Matches (resume_id, job_id, score) VALUES (?, ?, ?)",
            int(resume_id),
            int(job_id),
            float(scores[i][j])
        )

conn.commit()
cursor.close()
conn.close()

print("\n✅ Results saved successfully to SQL Server!")

