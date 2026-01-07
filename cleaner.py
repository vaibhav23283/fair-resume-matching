import re

def clean_resume(text):

    # Remove names at beginning (single or double word)
    text = re.sub(r"^[A-Z][a-z]+( [A-Z][a-z]+)?,?", "", text)

    # Remove numbers (age, years, etc.)
    text = re.sub(r"\b\d+\b", "", text)

    # Remove locations (expandable list)
    banned_words = ["India", "Delhi", "Bangalore", "Mumbai", "Chennai", "USA", "UK"]

    for word in banned_words:
        text = text.replace(word, "")

    # Clean extra commas and spaces
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"^\s*,", "", text)
    text = text.strip()

    return text
