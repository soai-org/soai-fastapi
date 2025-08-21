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
            device_map={"": "cpu"}  # GPU 사용 시 auto, CPU 사용 시 지정
        )
        
        # 2️⃣ LoRA adapter 로드
        model = PeftModel.from_pretrained(self.model, adapter_path)
        
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
    Gemma3_1b = FineTunedLLMModel()  # GPU 있으면 device="cuda"
    question = """다음 중 가족 중심 중재(Family-Centered Intervention)에 대한 설명으로 옳은 것은?  
                1) 가족 중심 중재는 모든 환자에서 동일한 효과를 보인다.  
                2) 가족 중심 중재는 부작용이 자주 발생한다.  
                3) 가족 중심 중재는 환자와 가족의 전반적인 건강과 복지를 향상시킬 가능성이 있다.  
                4) 가족 중심 중재는 신체 건강만을 개선하는 데 초점이 맞춰져 있다.  
                5) 가족 중심 중재는 연구에서 항상 비뚤림 위험이 없는 것으로 나타났다."""
    
    answer = Gemma3_1b.generate_answer(question)
    print(f"테스트 출력: \n{answer}")
