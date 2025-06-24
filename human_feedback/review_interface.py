import streamlit as st
import os
import json

# Absolute base path
BASE_PATH = "/content/drive/MyDrive/Softnerve_Submission"

# File mapping with full paths
version_files = {
    "chapter1_v1":  "/Users/chaitanya/Automated_book_pub_workflow/data/raw/chapter_1.txt",
    "chapter1_v2":  "/Users/chaitanya/Automated_book_pub_workflow/data/spun_chapter_1.txt",
    "chapter1_v3":  "/Users/chaitanya/Automated_book_pub_workflow/data/reviewed_chapter_1.txt",
    "chapter1_v4":  "/Users/chaitanya/Automated_book_pub_workflow/data/Final_chapter.txt",
}

# Feedback JSON file location
feedback_path = os.path.join("utils", "feedback_rewards.json")

# Load previous feedback
if os.path.exists(feedback_path):
    with open(feedback_path, "r") as f:
        feedback_data = json.load(f)
else:
    feedback_data = {}

# UI
st.title("📘 Human Feedback: Chapter Version Evaluator")
st.write("Rate each version on a scale from 0 to 5 based on quality, coherence, and relevance.")

for version_id, filepath in version_files.items():
    st.subheader(f"📄 {version_id} — {os.path.basename(filepath)}")
    
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            st.text_area("Content", content, height=250, disabled=True)
    else:
        st.warning(f"⚠️ File not found: {filepath}")
        continue

    current_score = feedback_data.get(version_id, 0.0)
    score = st.slider(f"Your Rating for {version_id}", 0.0, 5.0, float(current_score), 0.5)
    feedback_data[version_id] = score

if st.button("💾 Save Feedback"):
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    with open(feedback_path, "w") as f:
        json.dump(feedback_data, f, indent=2)
    st.success("✅ Feedback saved to utils/feedback_rewards.json")
