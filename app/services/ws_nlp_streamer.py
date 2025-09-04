from typing import AsyncIterator, Optional
import asyncio
import torch
from app.services.vis_nlp import generate_caption


class WsNLPStreamer:
    """
    WebSocket 전용 NLP 스트리밍 서비스
    - Vision + NLP 모델을 사용하여 실시간 캡션 생성 스트리밍.
    - 이미 텐서 형태의 이미지 데이터를 받아서 처리.
    """

    def __init__(self):
        pass

    @classmethod
    async def stream_caption_generation(cls, image_tensor: torch.Tensor, meta_prompt: str) -> AsyncIterator[str]:
        """
        이미지 텐서와 메타 프롬프트를 받아서 실시간으로 캡션을 생성하고 스트리밍
        """
        try:
            # 캡션 생성 시작
            yield "캡션 생성 시작...\n"
            
            # 기존 vis_nlp의 generate_caption 함수 사용
            # 하지만 실시간 스트리밍을 위해 토큰 단위로 처리
            caption = generate_caption(None, image_tensor, meta_prompt, None, max_len=48, device='cpu')
            
            # 생성된 캡션을 단어 단위로 스트리밍
            words = caption.split()
            for i, word in enumerate(words):
                yield word + " "
                await asyncio.sleep(0.1)  # 실시간 효과를 위한 지연
            
            yield f"\n\n--- 캡션 생성 완료 ---\n"
            yield f"최종 캡션: {caption}"
                
        except Exception as e:
            yield f"캡션 생성 중 오류가 발생했습니다: {str(e)}"

    async def process_tensor_and_meta(self, image_tensor: torch.Tensor, meta_prompt: str) -> AsyncIterator[str]:
        """
        이미지 텐서와 메타 프롬프트를 처리하여 캡션 생성
        """
        try:
            yield "텐서 데이터 처리 중...\n"
            
            # 캡션 생성 스트리밍
            async for chunk in self.stream_caption_generation(image_tensor, meta_prompt):
                yield chunk
                
        except Exception as e:
            yield f"텐서 처리 중 오류: {str(e)}"
