import re

#Define a list of known skills
KNOWN_SKILLS = [
    'python', 'java', 'c++', 'javascript', 'html', 'css',
    'machine learning', 'deep learning', 'data science',
    'nlp', 'sql', 'pandas', 'numpy', 'tensorflow', 'pytorch'
]

def extract_skills(text):
    """
    Extract known skills from user input text.
    """
    text = text.lower()
    extracted = []

    for skill in KNOWN_SKILLS:
        #Match full skill phrase using word boundaries
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            extracted.append(skill)

    return extracted


