# from openai import OpenAI

# client = OpenAI()

# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "Write a haiku about recursion in programming."
#         }
#     ]
# )

# print(completion.choices[0].message)

## needs subscription to OpenAI to run

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m"  # Smaller model for CPU efficiency
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate_response("who are you?"))
print(generate_response("what is the meaning of life?"))
print(generate_response("what is the weather like in San Francisco?"))