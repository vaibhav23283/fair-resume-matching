# Fair Resume–Job Matching System with Bias Reduction

## Problem
Traditional hiring systems often suffer from unconscious bias due to personal identifiers such as names, age, and location. 
This leads to unfair screening and missed talent.

## Solution
This project implements an AI-based resume–job matching system that:
- Removes bias indicators from resumes
- Uses semantic AI embeddings for skill-based matching
- Ranks candidates fairly based on similarity scores
- Stores results transparently for auditability

## Microsoft Technologies Used
- GitHub Models – AI embeddings
- SQL Server Express – Data storage
- Visual Studio Code – Development environment

## Responsible AI
- Personal identifiers are removed before processing
- Matching is based purely on skills and experience
- Match scores are stored transparently

## How to Run
```bash
pip install -r requirements.txt
python main.py
