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
    
    # 클래스 변수로 모델 인스턴스 저장 (싱글톤 패턴)
    _llm_model = None

    def __init__(self):
        # 모델 인스턴스 생성
        self.llm_model = FineTunedLLMModel()
        self.tokenizer = self.llm_model.tokenizer
        self.model = self.llm_model.model

    @classmethod
    def get_llm_model(cls):
        """모델 인스턴스를 싱글톤으로 관리"""
        if cls._llm_model is None:
            print("LLM 모델 로딩 시작...")
            try:
                cls._llm_model = FineTunedLLMModel()
                print("LLM 모델 로딩 완료!")
            except Exception as e:
                print(f"LLM 모델 로딩 실패: {e}")
                raise e
        return cls._llm_model
    
    @classmethod
    def preload_model(cls):
        if cls._llm_model is None:
            cls.get_llm_model()

    @classmethod
    async def stream_answer(cls, prompt: str) -> AsyncIterator[str]:
        try:
            # 싱글톤 모델 인스턴스 사용
            llm_model = cls.get_llm_model()
            
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
                    
        except Exception as e:
            yield f"오류가 발생했습니다: {str(e)}"


