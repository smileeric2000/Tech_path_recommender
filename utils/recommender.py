# recommend_paths.py

# A sample skill-to-learning-path mapping (expand as needed)
LEARNING_PATHS = {
    'python': [
        'Complete Python for Beginners on Coursera',
        'Automate the Boring Stuff with Python (book)'
    ],
    'data science': [
        'Data Science Specialization by Johns Hopkins (Coursera)',
        'Hands-on Machine Learning with Scikit-Learn and TensorFlow (book)'
    ],
    'machine learning': [
        'Machine Learning by Andrew Ng (Coursera)',
        'Hands-On Machine Learning with Scikit-Learn and TensorFlow (book)'
    ],
    'deep learning': [
        'Deep Learning Specialization (Coursera)',
        'Deep Learning with Python by François Chollet (book)'
    ],
    'pandas': [
        'Data Manipulation with pandas on DataCamp',
        'Pandas Documentation Tutorials'
    ],
    'nlp': [
        'Natural Language Processing with Python (NLTK book)',
        'Deep Learning for NLP with PyTorch (Coursera)'
    ]
}

# Optional: Define prerequisite relationships or skill progression paths
PREREQUISITES = {
    'machine learning': ['python', 'data science'],
    'deep learning': ['machine learning'],
    'nlp': ['machine learning', 'python']
}

def recommend_paths(extracted_skills):
    """
    Recommend personalized learning paths based on extracted skills.
    Suggest next skills to learn and resources for both current and next skills.
    """
    recommendations = {}
    next_skills = set()

    # Include paths for skills user already knows
    for skill in extracted_skills:
        if skill in LEARNING_PATHS:
            recommendations[skill] = LEARNING_PATHS[skill]

        # Suggest next skills based on prerequisites mapping
        for s, prereqs in PREREQUISITES.items():
            if skill in prereqs and s not in extracted_skills:
                next_skills.add(s)

    # Add recommendations for next skills user can learn
    for skill in next_skills:
        if skill in LEARNING_PATHS:
            recommendations[skill] = LEARNING_PATHS[skill]

    return recommendations
