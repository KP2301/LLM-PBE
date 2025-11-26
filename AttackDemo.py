# ใช้งาน Groq
from google.colab import userdata
import os
import sys

# เพิ่ม path
sys.path.insert(0, '/content/LLM-PBE')

# ตั้งค่า API key (ฟรี - สมัครที่ https://console.groq.com)
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')

from data import JailbreakQueries
from models.GroqModels import GroqModels
from attacks import Jailbreak
from metrics import JailbreakRate

data = JailbreakQueries()

# โมเดลฟรีที่เร็วมาก
llm = GroqModels(
    model="llama-3.1-8b-instant"  # หรือ "mixtral-8x7b-32768" หรือ "gemma2-9b-it"
)

attack = Jailbreak()
results = attack.execute_attack(data, llm)

rate = JailbreakRate(results).compute_metric()
print(f"Model: llama-3.1-8b-instant (Groq)")
print(f"Jailbreak Rate: {rate}")


# from data import JailbreakQueries
# from models import TogetherAIModels
# from attacks import Jailbreak
# from metrics import JailbreakRate

# data = JailbreakQueries()
# # Fill api_key
# llm = TogetherAIModels(model="togethercomputer/llama-2-7b-chat", api_key="")
# attack = Jailbreak()
# results = attack.execute_attack(data, llm)
# rate = JailbreakRate(results).compute_metric()
# print("rate:", rate)
