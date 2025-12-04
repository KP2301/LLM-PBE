#--- Data Extraction Attack ---

from google.colab import userdata
import os
import sys

sys.path.insert(0, '/content/LLM-PBE')

os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')

from models.GroqModels import GroqModels
from attacks.DataExtraction.enron import EnronDataExtraction
from attacks.DataExtraction.prompt_extract import PromptExtraction

enron = EnronDataExtraction(data_path="data/enron")
atk_format = '3-shot-known-domain-c'
prompts, _ = enron.generate_prompts(format=atk_format)

llm_model = "llama-3.1-8b-instant"
llm = GroqModels(model=llm_model)
attack = PromptExtraction()
results = attack.execute_attack(prompts[:100], llm)

print("="*100)
print(f"DATA EXTRACTION RESULTS - Total: {len(results)} samples")
print("="*100)

for idx in range(len(results)):
    print(f"\n{'─'*100}")
    print(f"SAMPLE #{idx + 1}")
    print(f"{'─'*100}")

    print(f"\n📝 PROMPT:")
    print(prompts[idx])

    print(f"\n💬 ANSWER:")
    print(results[idx])
    print()

print("="*100)