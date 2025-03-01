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

from structures import Chatbot

model_name = "facebook/opt-125m"  # Smaller model for CPU efficiency
chat = Chatbot(model_name)

print(chat.generate_response("Hello, how are you?"))

# Define how the output should look like

# print(chat.generate_response_with_structure("what is the weather like in New York?"))