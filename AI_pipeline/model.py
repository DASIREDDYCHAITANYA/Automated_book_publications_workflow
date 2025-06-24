from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
model = AutoModelForCausalLM.from_pretrained(model_name, token="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")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
