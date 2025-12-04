#--- Jailbreak Attack Demo ---

# ใช้งาน Groq
from google.colab import userdata
import os
import sys

# เพิ่ม path
sys.path.insert(0, '/content/LLM-PBE')

# name: llm-pbe , api-key: xxxx
# ตั้งค่า API key (ฟรี - สมัครที่ https://console.groq.com)
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')

from models.GroqModels import GroqModels
from data import JailbreakQueries
from attacks import Jailbreak
from metrics import JailbreakRate

data = JailbreakQueries()
data.generate_queries()

llm = GroqModels(
    model="llama-3.1-8b-instant"  # or "mixtral-8x7b-32768" or "gemma2-9b-it"
)

attack = Jailbreak()
results = attack.execute_attack(data, llm)
rate = JailbreakRate(results).compute_metric()
print(f"Model: llama-3.1-8b-instant (Groq)")
print(f"Jailbreak Rate: {rate}")
