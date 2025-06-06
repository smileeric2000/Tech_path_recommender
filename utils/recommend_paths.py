import json
#A sample skill-to-learning-path mapping (expand as needed)
def load_learning_paths(json_file="learning_paths.json"):
    with open(json_file, "r") as f:
        return json.load(f)


def recommend_paths(extracted_skills, json_file="data/learning_paths.json"):
     
   
    """
    Recommend personalized learning paths with:
    - resources for known and next skills,
    - unlearned skills per path,
    - time estimates,
    - resource links.
    
    Returns a dictionary with each skill mapped to details.
    """
    learning_paths = load_learning_paths(json_file)
    recommendations = {}
    next_skills = set()

    # Suggest next skills based on prerequisites
    for skill in extracted_skills:
        for s, details in learning_paths.items():
            prereqs = details.get("prerequisites", [])
            if skill in prereqs and s not in extracted_skills:
                next_skills.add(s)

    all_skills_to_recommend = set(extracted_skills) | next_skills

    for skill in all_skills_to_recommend:
        if skill in learning_paths:
            unlearned = [
                pre for pre in learning_paths[skill].get("prerequisites", [])
                if pre not in extracted_skills
            ]

            recommendations[skill] = {
                "resources": learning_paths[skill].get("resources", []),
                "unlearned_prerequisites": unlearned
            }

    return recommendations