import requests
import openai
from fpdf import FPDF
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Make sure your OPENAI_API_KEY is set as an environment variable outside this module

# openai.api_key = os.getenv("OPENAI_API_KEY")


def fetch_coursera_courses(query, limit=5):
    """
    Fetch top courses from Coursera matching the query.
    """
    if not query:
        return []
    url = "https://api.coursera.org/api/courses.v1"
    params = {
        "q": "search",
        "query": query,
        "limit": limit,
        "fields": "name,description,slug"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  #will raise an HTTPerror for bad reponses
        data = response.json()

        courses = []
        for course in data.get("elements", []):
            name = course.get("name")
            slug = course.get("slug")
            url = f"https://www.coursera.org/learn/{slug}" if slug else ""
            courses.append({"name": name, "url": url})
        return courses
    
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Coursera API request failed: {e}")
        return []


def generate_course_recommendations(user_skills, learning_path, experience_level):
    """
    Use OpenAI GPT to generate course recommendations.
    """
    model_path = r"C:\Users\USER\data_science\Classes_with_SirPsalms\Deep_learning\Transformer_based_NLP\Project_Pulse\my_fine_tuned_model"

    prompt = f"""
    You are a helpful learning advisor.

    User has skills: {', '.join(user_skills)}.
    They want to learn: {learning_path}.
    Their experience level is: {experience_level}.

    Suggest 3 online courses (with titles) suitable for this user on platforms like Coursera, EdX, or Udemy.
    Provide short descriptions for each course.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=250, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def create_pdf_report(user_skills, experience_level, recommended_paths, course_recommendations):
    """
    Generate a PDF report summarizing the user skills, experience level,
    recommended learning paths, and course recommendations.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Personalized Learning Path Report", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Skills: {', '.join(user_skills)}", ln=True)
    pdf.cell(0, 10, f"Experience Level: {experience_level}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Recommended Learning Paths:", ln=True)
    pdf.set_font("Arial", "", 12)
    for path in recommended_paths:
        pdf.cell(0, 10, f"- {path}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Course Recommendations:", ln=True)
    pdf.set_font("Arial", "", 12)
    for line in course_recommendations.split("\n"):
        pdf.cell(0, 10, line, ln=True)

    filename = "learning_path_report.pdf"
    pdf.output(filename)
    return filename
