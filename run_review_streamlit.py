# run_review_streamlit.py

import os

# Option 1: Use full absolute path (for Colab or Drive)
streamlit_script_path = "/human_feedback/review_interface.py"

# Option 2: Or use relative path
# streamlit_script_path = os.path.join("human_feedback", "review_interface.py")

os.system(f"streamlit run {streamlit_script_path} --server.port 8501")
