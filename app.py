import streamlit as st
import re
import json
import requests
import os
import openai
from pdf_generator import create_pdf_report  # from previous snippet or define inline
from pdf_generator import fetch_coursera_courses, generate_course_recommendations

openai.api_key = os.getenv("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")  # Set this in your environment

# Load learning paths
@st.cache_data
def load_learning_paths():
    with open("data/learning_paths.json", "r") as f:
        return json.load(f)

def extract_skills_from_text(text, keywords):
    found_skills = [kw for kw in keywords if re.search(rf"\b{kw}\b", text, re.IGNORECASE)]
    return found_skills

def main():
    st.title("📘 Personalized Learning Path Recommender with Course Suggestions")

    learning_paths = load_learning_paths()
    all_skills = sorted({skill for skills in learning_paths.values() for skill in skills})

    # User inputs
    input_type = st.radio("Choose input type", ["Upload Resume", "Select Skills"])
    experience_level = st.selectbox("Select your experience level", ["Beginner", "Intermediate", "Advanced"])

    user_skills = []

    if input_type == "Upload Resume":
        uploaded_file = st.file_uploader("Upload your resume (.txt only for demo)", type=["txt"])
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
            user_skills = extract_skills_from_text(text, all_skills)
            st.write("### Extracted Skills:")
            if user_skills:
                st.write(user_skills)
            else:
                st.write("No known skills detected. Try manual selection.")

    elif input_type == "Select Skills":
        user_skills = st.multiselect("Select your current skills", all_skills)

    if user_skills:
        # Recommend learning paths by skill overlap
        scores = {}
        for path, skills in learning_paths.items():
            overlap = set(user_skills).intersection(set(skills))
            scores[path] = len(overlap) / len(skills)
        recommended_paths = sorted(scores, key=scores.get, reverse=True)

        st.write("### 🎯 Recommended Learning Paths:")
        for path in recommended_paths:
            st.markdown(f"**{path}** — Skills: {', '.join(learning_paths[path])}")

        # Use OpenAI GPT to generate personalized course recommendations
        if st.button("Get Course Recommendations"):
            with st.spinner("Generating course recommendations..."):
                course_recs = generate_course_recommendations(user_skills, recommended_paths[0], experience_level)
                st.markdown("### 📚 Course Recommendations:")
                st.write(course_recs)

                # Fetch some real courses from Coursera based on first recommended path
                courses = fetch_coursera_courses(recommended_paths[0])
                st.markdown("### 🔎 Real Coursera Courses:")
                for course in courses:
                    st.markdown(f"- [{course['name']}]({course['url']})")

                # PDF export
                if st.button("Download PDF Report"):
                    filename = create_pdf_report(user_skills, experience_level, recommended_paths, course_recs)
                    with open(filename, "rb") as f:
                        st.download_button(label="Download PDF", data=f, file_name=filename, mime="application/pdf")

if __name__ == "__main__":
    main()
