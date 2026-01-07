# -----------------------------
# IMPORTS
# -----------------------------
import pandas as pd
import numpy as np
import pyodbc
from openai import OpenAI
from cleaner import clean_resume

# -----------------------------
# MICROSOFT AI (GitHub Models)
# -----------------------------
GITHUB_TOKEN = "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"

client = OpenAI(
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com"
)

def get_embedding(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

# -----------------------------
# LOAD DATA
# -----------------------------
resumes = pd.read_csv("resumes.csv")
jobs = pd.read_csv("jobs.csv")

# -----------------------------
# BIAS REMOVAL (ANONYMIZATION)
# -----------------------------
resumes["text"] = resumes["text"].apply(clean_resume)

# -----------------------------
# GENERATE EMBEDDINGS
# -----------------------------
resume_vectors = np.array([get_embedding(t) for t in resumes["text"]])
job_vectors = np.array([get_embedding(t) for t in jobs["description"]])

# -----------------------------
# COSINE SIMILARITY
# -----------------------------
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

scores = []
for r in resume_vectors:
    row = []
    for j in job_vectors:
        row.append(cosine_similarity(r, j))
    scores.append(row)

# -----------------------------
# PRINT RANKING (CONSOLE)
# -----------------------------
print("\n=== FAIR RESUME–JOB MATCHING RESULTS ===\n")

for j, job in jobs.iterrows():
    print(f"Job {job['id']}: {job['description']}")

    job_scores = []
    for i, resume in resumes.iterrows():
        score = scores[i][j]
        job_scores.append((resume["text"], score))

    job_scores.sort(key=lambda x: x[1], reverse=True)

    for rank, (text, score) in enumerate(job_scores[:2], start=1):
        print(f"   Rank {rank}: {text}")
        print(f"      Match Score: {score:.3f}")

    print("-" * 50)

# -----------------------------
# SAVE RESULTS TO SQL SERVER
# -----------------------------
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=localhost\\SQLEXPRESS;"
    "Database=ResumeMatchingDB;"
    "Trusted_Connection=yes;"
)

cursor = conn.cursor()

# Clear old results
cursor.execute("DELETE FROM Matches")
conn.commit()

# Insert new scores
for j, job in jobs.iterrows():
    for i, resume in resumes.iterrows():
        cursor.execute(
            "INSERT INTO Matches (resume_id, job_id, score) VALUES (?, ?, ?)",
            int(i + 1), int(j + 1), float(scores[i][j])
        )

conn.commit()
cursor.close()
conn.close()

print("\n✅ Results saved successfully to SQL Server!")
