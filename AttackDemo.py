# Groq
from google.colab import userdata
import os
import sys

# add path
sys.path.insert(0, '/content/LLM-PBE')

# set API key (free - register at https://console.groq.com)
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')

from models.GroqModels import GroqModels

#--- Jailbreak Attack Demo ---
from data import JailbreakQueries
from attacks import Jailbreak
from metrics import JailbreakRate

data = JailbreakQueries()

llm = GroqModels(model="llama-3.1-8b-instant")  # or "mixtral-8x7b-32768" or "gemma2-9b-it"

attack = Jailbreak()
results = attack.execute_attack(data, llm)

rate = JailbreakRate(results).compute_metric()
print(f"Model: llama-3.1-8b-instant (Groq)")
print(f"Jailbreak Rate: {rate}")

#--- Data Extraction Attack Demo ---
from attacks.DataExtraction.enron import EnronDataExtraction
from attacks.DataExtraction.prompt_extract import PromptExtraction

enron = EnronDataExtraction(data_path="data/enron")

# for format in ['prefix-50','0-shot-known-domain-b','0-shot-unknown-domain-c', '3-shot-known-domain-c', '5-shot-unknown-domain-b']:
#     prompts, _ = enron.generate_prompts(format=format)
#     print("prompts: ", prompts)
#     llm = GroqModels(model="llama-3.1-8b-instant")
#     attack = PromptExtraction()
#     results = attack.execute_attack(prompts, llm)
#     print("results:", results)

prompts, labels = enron.generate_prompts(format='3-shot-known-domain-c')
print("Type of prompts:", type(prompts))
print("Number of prompts:", len(prompts[:10]) if isinstance(prompts, list) else "NOT A LIST!")
print("First prompt:", prompts[0] if isinstance(prompts, list) else prompts[:50])

llm = GroqModels(model="llama-3.1-8b-instant")
attack = PromptExtraction()
results = attack.execute_attack(prompts[:10], llm)
print("results:", results)