#Import necessary libraries and dependencies
import streamlit as st
from utils.parser import extract_skills
from utils.recommender import recommend_paths
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pdf_generator import create_pdf_report, fetch_coursera_courses, generate_course_recommendations
from datasets import load_dataset
import json
import requests
from fpdf import FPDF



#!/usr/bin/env python
# coding: utf-8

# # Personalized learning path 
# 
# _This model/app allows Users to uploads their resume or selects current skills → app recommends personalized learning paths_

# **Tools**
# 
# 🧠 NLP (for resume/skills parsing)
# 
# 🎯 Similarity matching (cosine similarity or embedding-based)
# 
# 📊 Streamlit (frontend)
# 
# 📚 Predefined learning paths dataset (you'll create this)
# 
# 🔍 Optional: LLM or Flan-T5 for suggesting learning goals
#learning_recommender/
#│
#├── app.py                   # Main Streamlit app
#├── data/
#│   └── learning_paths.json  # Skills mapped to paths
#├── utils/
#│   └── parser.py            # Resume parser
#│   └── recommender.py       # Recommender logic
#├── sample_resume.txt
#├── requirements.txt
#└── README.md

# #### Define Learning Paths

# In[1]:



text = {
  "python": {
    "resources": [
      {
        "title": "Complete Python for Beginners",
        "platform": "Coursera",
        "time_estimate": "20 hours",
        "link": "https://www.coursera.org/learn/python"
      },
      {
        "title": "Automate the Boring Stuff with Python",
        "platform": "Book",
        "time_estimate": "15 hours",
        "link": "https://automatetheboringstuff.com/"
      }
    ],
    "prerequisites": []
  },
  "machine learning": {
    "resources": [
      {
        "title": "Machine Learning by Andrew Ng",
        "platform": "Coursera",
        "time_estimate": "55 hours",
        "link": "https://www.coursera.org/learn/machine-learning"
      },
      {
        "title": "Hands-On Machine Learning",
        "platform": "Book",
        "time_estimate": "40 hours",
        "link": "https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/"
      }
    ],
    "prerequisites": ["python", "data science"]
  }
}


# Save to JSON
# with open("learning_paths.json", "w") as f:
#     json.dump(text, f, indent=2)


# #### Step 2 Resume or Skill Input
# 
# * Define Resume upload function (Allow Resume to be of types-> (.txt or .pdf)
# 
# * Manual Skill Selection (multi-select box)

# In[2]:






# NOTE :
# 
#     utils → package (folder with __init__.py)    
#     parser.py → module (a single Python file inside utils)
#     extract_skills → function (defined inside the parser.py module)
# 
# The "extract_skills" — in utils/parser.py
# Parses a given text (e.g., user input) and extracts known skills mentioned in it.
# How it works:
# 
#     Checks the input text against a predefined list of skills and returns all the skills found.
#     
# The "recommend_paths" — in recommender.py
# Takes a list of skills the user already knows and recommends personalized learning paths and resources.
# How it works:
# 
#     For each known skill, it provides relevant learning resources.
#     Identifies “next skills” the user should learn based on prerequisite relationships and recommends resources for those too.
# 
# Output:
# A dictionary mapping each skill (both current and recommended next skills) to a list of learning resources (e.g., courses, books).
# 
# 

# In[9]:


st.title("📘 Personalized Learning Path Recommender")

skills = []  #Initialize skills <<empty list of skills>>

option = st.radio("Choose Input Type", ["Upload Resume", "Select Skills"])

if option == "Upload Resume":
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt"])

    if uploaded_file:
        #Read file bytes and decode to string
        text = uploaded_file.read().decode("latin-1")
        #Pass the text content
        skills = extract_skills(text)
        st.write("**Extracted Skills:**", skills)

elif option == "Select Skills":
    all_skills = ["Python", "SQL", "Pandas", "Scikit-learn", "Spark", "Airflow", "Numpy", "Transformers"]
    skills = st.multiselect("Select your current skills", all_skills)

if skills:
    recommendations = recommend_paths(skills)
    st.write("### 🎯 Recommended Paths:")
    for rec in recommendations:
        st.markdown(f"- **{rec}**")


# #### Step 3 Extract Skills (NLP Resume Parser)

# #### Step 4 Recommend Based on Skill Overlap

# In[10]:



def recommend_paths(user_skills):
    with open("data/learning_paths.json", "r") as f:
        paths = json.load(f)

    scores = {}
    for path, details in paths.items():
        skill_list = details.get("prerequisites", [])
        overlap = set(user_skills).intersection(set(skill_list))
        scores[path] = len(overlap) / len(skill_list) if skill_list else 0

    return sorted(scores, key=scores.get, reverse=True)


# #### Step 6: (Optional) Suggest Next Skills to Learn
# 
# Extend recommend_paths() to also return:
#     
#     Unlearned skills per path
#     Time estimates
#     Resources (links from Medium, YouTube, etc.)
# 

# #### Step 7 LLM Integration ( Skill-to-Course Match Using APIs)
# 
#     Using HuggingFace’s OpenAI GPT API.
# 
#     Fetch real-time course data from platforms like Coursera, EdX, Udemy.
# 
# 
#     
#     Input: user skills + recommended learning path.
#
#     Output: list of course names + short description.

###### Coursera API fetch function



# def fetch_coursera_courses(query, limit=5):
#     """
#     Fetch top courses from Coursera matching the query.
#     """
#     url = "https://api.coursera.org/api/courses.v1"
#     params = {
#         "q": "search",
#         "query": query,
#         "limit": limit,
#         "fields": "name,description,slug"
#     }
#     response = requests.get(url, params=params)
#     data = response.json()

#     courses = []
#     for course in data.get("elements", []):
#         name = course.get("name")
#         slug = course.get("slug")
#         url = f"https://www.coursera.org/learn/{slug}" if slug else ""
#         courses.append({"name": name, "url": url})
#     return courses


#AI Integration

def generate_course_recommendations(user_skills, learning_path, experience_level):
    return generate_course_recommendations(user_skills, learning_path, experience_level)




# def create_pdf_report(user_skills, experience_level, recommended_paths, course_recommendations):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(0, 10, "Personalized Learning Path Report", ln=True)
#     pdf.set_font("Arial", "", 12)
#     pdf.cell(0, 10, f"Skills: {', '.join(user_skills)}", ln=True)
#     pdf.cell(0, 10, f"Experience Level: {experience_level}", ln=True)
#     pdf.ln(10)

#     pdf.set_font("Arial", "B", 14)
#     pdf.cell(0, 10, "Recommended Learning Paths:", ln=True)
#     pdf.set_font("Arial", "", 12)
#     for path in recommended_paths:
#         pdf.cell(0, 10, f"- {path}", ln=True)

#     pdf.ln(10)
#     pdf.set_font("Arial", "B", 14)
#     pdf.cell(0, 10, "Course Recommendations:", ln=True)
#     pdf.set_font("Arial", "", 12)
#     for line in course_recommendations.split("\n"):
#         pdf.cell(0, 10, line, ln=True)

#     filename = "learning_path_report.pdf"
#     pdf.output(filename)
#     return filename


###  Streamlit Full App Example (app.py)

# In[14]:

model_path = r"C:\Users\USER\data_science\Classes_with_SirPsalms\Deep_learning\Transformer_based_NLP\Project_Pulse\my_fine_tuned_model"

#Define torch device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_gpt2_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    #PAD TOKENS
    tokenizer.pad_token = tokenizer.eos_token  

    model.eval()
    return tokenizer, model

tokenizer, model = load_gpt2_model()

@st.cache_data
def load_learning_paths():
    with open("data/learning_paths.json", "r") as f:
        return json.load(f)

def extract_skills_from_text(text, keywords):
    return [kw for kw in keywords if re.search(rf"\b{kw}\b", text, re.IGNORECASE)]

def gpt2_generate(prompt, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", padding = True).to(device)
    st.write("This is before GPT text")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length = max_length,
            do_sample = True,
            top_k=50,
            top_p=0.95,
            temperature=0.75,
            pad_token_id=tokenizer.eos_token_id
        )
    
        #decode bot reponse
        #DEBUGGER
        raw_generated = tokenizer.decode(output[0], skip_special_tokens=True)
        #DEBUGGER
        #st.write("Raw GPT output:", raw_generated)  # Debug print

        
        # Process the generated text: remove the prompt portion and anything after "User:"
        processed = raw_generated[len(prompt):].strip().split("User:")[0].strip()
        st.write("Processed GPT output:", processed)  # Debug print
        st.write("This is after GPT response")
        return processed
        # generated = tokenizer.decode(output[0], skip_special_tokens=True)
        # return generated[len(prompt):].strip().split("User:")[0].strip()

def main():
    st.title("📘 Personalized Learning Path Recommender")

    learning_paths = load_learning_paths()
    all_skills = sorted({skill for skills in learning_paths.values() for skill in skills})

    input_type = st.radio("Choose input type", ["Upload Resume", "Select Skills"])
    experience_level = st.selectbox("Select your experience level", ["Beginner", "Intermediate", "Advanced"])

    user_skills = []

    if input_type == "Upload Resume":
        uploaded_file = st.file_uploader("Upload your resume (.txt only)", type=["txt"])
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
            user_skills = extract_skills_from_text(text, all_skills)
            st.write("### Extracted Skills:")
            st.write(user_skills if user_skills else "No known skills detected. Try manual selection.")

    elif input_type == "Select Skills":
        user_skills = st.multiselect("Select your current skills", all_skills)

    if user_skills:
        scores = {}
        for path, skills in learning_paths.items():
            overlap = set(user_skills).intersection(skills)
            scores[path] = len(overlap) / len(skills)
        recommended_paths = sorted(scores, key=scores.get, reverse=True)

        st.write("### 🎯 Recommended Learning Paths:")
        for path in recommended_paths:
            st.markdown(f"**{path}** — Skills: {', '.join(learning_paths[path])}")

        if st.button("Get Course Recommendations"):
            with st.spinner("Generating personalized suggestions..."):
                prompt = (
                    f"Suggest personalized courses for someone with the following skills: {', '.join(user_skills)}. "
                    f"The recommended learning path is: {recommended_paths[0]}. "
                    f"Their experience level is {experience_level}.\nCourses:"
                )
                course_recs = gpt2_generate(prompt)

                st.markdown("### 📚 Course Recommendations:")
                st.write(f"GPT response :{course_recs}")  #Part to output GPT's resonse

                courses = fetch_coursera_courses(recommended_paths[0])
                st.markdown("### 🔎 Real Coursera Courses:")
                for course in courses:
                    st.markdown(f"- [{course['name']}]({course['url']})")

                if st.button("Download PDF Report"):
                    filename = create_pdf_report(user_skills, experience_level, recommended_paths, course_recs)
                    with open(filename, "rb") as f:
                        st.download_button(label="Download PDF", data=f, file_name=filename, mime="application/pdf")

if __name__ == "__main__":
    main()


# In[ ]:




