from utils.parser import extract_skills

user_input = "I have experience with Python, pandas, and deep learning."
skills = extract_skills(user_input)

print("Extracted skills:", skills)
