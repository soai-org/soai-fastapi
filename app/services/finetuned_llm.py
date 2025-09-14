from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import os
from dotenv import load_dotenv

class FineTunedLLMModel:
    def __init__(self, base_model_path="google/gemma-3-1b-it", adapter_path="app/services/gemma-3-1b-it-finetuned-final"):
        
        load_dotenv()
        HF_TOKEN = os.getenv("HF_TOKEN")
        login(HF_TOKEN)
        
        # 1️⃣ Base 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        # 2️⃣ LoRA adapter 로드
        model = PeftModel.from_pretrained(self.model, 
                                          adapter_path,        
                                          strict=False) # 레이어 불일치 경고 무시하고 로드)
        
        # 3️⃣ LoRA + Base 모델 병합
        model = model.merge_and_unload()
        model = model.to('cpu')
        
        # 4️⃣ 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # 5️⃣ 프롬프트 템플릿
        self.prompt_template = (
            "### 질문: {question}\n"
            "### 조건:\n"
            "- 답변은 최대 7줄까지만 작성\n"
            "- 전문적이면서도 이해하기 쉽게 작성\n"
            "- 필요한 경우 예시 포함\n"
            "- 한글로만 답변할 것\n"
            "### 답변: "
        )
        self.model = model

    def generate_answer(self, user_input, max_new_tokens=250):
        # 프롬프트 생성
        prompt = self.prompt_template.format(question=user_input)
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cpu')
        
        # 모델에서 텍스트 생성
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded_output.split("### 답변: ")[-1].strip()
        return answer

# ------------------------------
# 사용 예시
if __name__ == "__main__":
    pass