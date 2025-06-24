from model import model,tokenizer
def build_review_prompt(spun_text):
    return f"""You are an expert editor. Review the following rewritten passage for clarity, grammar, and flow.
Suggest improvements, but do not change the meaning.

Text to review:
\"\"\"
{spun_text}
\"\"\"

Reviewed and improved version:
"""

# Step 1: Load the AI-spun text
with open("/data/spun_chapter_1.txt", "r", encoding="utf-8") as f:
    spun_text = f.read()

# Step 2: Generate AI review
review_prompt = build_review_prompt(spun_text)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

review_output = generator(review_prompt, max_new_tokens=500, do_sample=True, temperature=0.7)
reviewed_text = review_output[0]["generated_text"]

# Step 3: Save AI-reviewed version
with open("/data/reviewed_chapter_1.txt", "w", encoding="utf-8") as f:
    f.write(reviewed_text)

print("✅ Reviewed chapter saved to data/reviewed_chapter_1.txt")
