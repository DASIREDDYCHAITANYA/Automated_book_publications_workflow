from model import model,tokenizer
def build_prompt(text):
    return f"Rewrite the following passage in modern, simplified English:\n\n{text}\n\nRewritten Version:\n"

# Step 3: Load the chapter text from file
with open("/data/raw/chapter_1.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Optional: Truncate text if too long (phi-2 has a context limit of ~2048 tokens)
short_text = raw_text[:1500]

# Step 4: Generate rewritten version
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = build_prompt(short_text)

output = pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.7)
rewritten_text = output[0]['generated_text']

# Step 5: Save to file
with open("/data/spun_chapter_1.txt", "w", encoding="utf-8") as f:
    f.write(rewritten_text)

print("✅ Rewritten chapter saved to /data/spun_chapter_1.txt")
