### 1st Block Code

!git clone https://github.com/KP2301/LLM-PBE
%cd LLM-PBE
!pip install --upgrade pip
!pip install git+https://github.com/microsoft/analysing_pii_leakage.git
!pip install wandb accelerate
!pip install -r requirements.txt
!pip install -q groq

================================================================================================

### 2nd Block Code

groq_code = '''
        import os
        from groq import Groq
        from models.LLMBase import LLMBase
        class GroqModels(LLMBase):
        def __init__(self, api_key=None, model="llama-3.1-8b-instant", max_tokens=256, temperature=0.7):
        super().__init__(api_key=api_key)
                if api_key:
                    self.client = Groq(api_key=api_key)
                elif "GROQ_API_KEY" in os.environ:
                    self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
                else:
                    self.client = Groq()
                self.model = model
                self.max_tokens = max_tokens
                self.temperature = temperature
                self.tokenizer = None
            def load_model(self):
                pass
            def query(self, prompt):
                return self.query_remote_model(prompt)
            def __call__(self, prompt):
                return self.query_remote_model(prompt)
            def query_remote_model(self, prompt):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"Groq API Error: {e}")
                    return ""
'''

with open('/content/LLM-PBE/models/GroqModels.py', 'w') as f:
f.write(groq_code)

print("✅ สร้าง GroqModels.py เรียบร้อย")

================================================================================================

### Then you can use dea.py and ja.py for 3rd and 4th block code
