from typing import AsyncIterator, Optional
import asyncio
import random
from app.services.finetuned_llm import FineTunedLLMModel
from transformers import TextIteratorStreamer
import threading


class WsLLMStreamer:
    """
    WebSocket 전용 LLM 스트리밍 서비스
    - FineTuned Gemma 모델을 사용하여 실시간 토큰 스트리밍.
    """

    def __init__(self):
        # 모델 인스턴스 생성
        self.llm_model = FineTunedLLMModel()
        self.tokenizer = self.llm_model.tokenizer
        self.model = self.llm_model.model

    @classmethod
    async def stream_answer(cls, prompt: str) -> AsyncIterator[str]:
        try:
            llm_model = FineTunedLLMModel()
            
            formatted_prompt = llm_model.prompt_template.format(question=prompt)
            inputs = llm_model.tokenizer(formatted_prompt, return_tensors="pt").to("cpu")
            
            streamer = TextIteratorStreamer(llm_model.tokenizer, skip_special_tokens=True)
            
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=250,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.2,
                pad_token_id=llm_model.tokenizer.eos_token_id,
                eos_token_id=llm_model.tokenizer.eos_token_id
            )
            
            def run_model_generation():
                return llm_model.model.generate(**generation_kwargs)
            
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, run_model_generation)
            
            for token in streamer:
                if token.strip(): 
                    yield token
                    # await asyncio.sleep(0.1)  # 실시간 효과를 위한 지연
                    
        except Exception as e:
            yield f"오류가 발생했습니다: {str(e)}"


