from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
