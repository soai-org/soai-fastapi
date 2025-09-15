from typing import AsyncIterator, Optional
import asyncio
import torch
from app.services.vis_nlp import generate_caption
from app.services.vis_nlp import VISNLPEXTRACTOR, CaptionGenerator, load_Generator
from transformers import AutoTokenizer


class WsNLPStreamer:
    """
    WebSocket 전용 NLP 스트리밍 서비스
    - Vision + NLP 모델을 사용하여 실시간 캡션 생성 스트리밍.
    - 이미 텐서 형태의 이미지 데이터를 받아서 처리.
    """
    
    # 클래스 변수로 모델 인스턴스 저장 (싱글톤 패턴)
    _nlp_model = None

    def __init__(self):
        pass

    @classmethod
    def get_nlp_model(cls):
        """모델 인스턴스를 싱글톤으로 관리"""
        if cls._nlp_model is None:
            print("NLP 모델 로딩 시작...")
            try:
                # 모델 구성 요소 초기화
                visnlp_extractor = VISNLPEXTRACTOR(
                    img_dim_h=512, img_dim_w=512, patch_size=16, 
                    embed_dim=256, num_heads=4, depth=4
                )
                
                # 토크나이저 로드 (BERT 기반)
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                
                # 캡션 생성기 초기화
                caption_generator = CaptionGenerator(
                    visnlp_extractor=visnlp_extractor,
                    vocab_size=tokenizer.vocab_size,
                    embed_dim=256,
                    num_heads=4,
                    ff_dim=2048,
                    num_layers=4
                )
                
                # 모델 로드
                model = load_Generator(caption_generator, 'app/models/image_meta.pth')
                
                # 모델과 토크나이저를 함께 저장
                cls._nlp_model = {
                    'model': model,
                    'tokenizer': tokenizer
                }
                
                print("NLP 모델 로딩 완료!")
            except Exception as e:
                print(f"NLP 모델 로딩 실패: {e}")
                raise e
        return cls._nlp_model
    
    @classmethod
    def preload_model(cls):
        """애플리케이션 시작 시 모델을 미리 로드"""
        if cls._nlp_model is None:
            print("🚀 애플리케이션 시작 시 NLP 모델 사전 로딩...")
            cls.get_nlp_model()
            print("✅ NLP 모델 사전 로딩 완료!")

    @classmethod
    async def stream_caption_generation(cls, image_tensor: torch.Tensor, meta_prompt: str) -> AsyncIterator[str]:
        """
        이미지 텐서와 메타 프롬프트를 받아서 실시간으로 캡션을 생성하고 스트리밍
        generate_caption을 대체하는 토큰 단위 스트리밍 버전
        """
        try:
            # 싱글톤 모델 인스턴스 사용 (미리 로드된 모델 즉시 사용)
            nlp_model = cls.get_nlp_model()
            
            # 실시간 토큰 스트리밍 (generate_caption을 대체)
            async for token in cls._generate_caption_stream(nlp_model, image_tensor, meta_prompt, max_len=48, device='cpu'):
                yield token
                    
        except Exception as e:
            yield f"캡션 생성 중 오류가 발생했습니다: {str(e)}"
    
    @classmethod
    async def _generate_caption_stream(cls, model_dict, image, meta, max_len=48, device='cpu'):
        """
        실시간으로 토큰을 생성하면서 스트리밍하는 내부 메서드
        vis_nlp.py의 generate_caption 함수를 진짜 토큰 단위 스트리밍 버전으로 구현
        """
        import asyncio
        
        model = model_dict['model']
        tokenizer = model_dict['tokenizer']
        
        model.eval()
        image = image.to(device)       
        meta = [meta]                                 

        with torch.no_grad():
            # 1. 이미지와 메타데이터에서 메모리 추출
            memory = model.extractor(image, meta).to(device)    
            
            # 2. 시작 토큰으로 초기화
            input_ids = torch.tensor([[tokenizer.cls_token_id]], device=device) 
            
            # 3. 토큰별로 실시간 생성 및 스트리밍
            for step in range(max_len - 1):
                # 다음 토큰 예측
                logits = model.decoder(input_ids, memory) 
                next_token_logits = logits[:, -1, :]      
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # 토큰을 텍스트로 변환
                token_text = tokenizer.decode(next_token, skip_special_tokens=True)
                
                # 특수 토큰이 아닌 경우에만 스트리밍
                if (next_token.item() not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] 
                    and token_text.strip()):  # 빈 토큰도 제외
                    yield token_text + " "
                
                # 다음 반복을 위해 input_ids 업데이트
                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

                # 종료 토큰이면 중단
                if next_token.item() == tokenizer.sep_token_id:
                    break
                    
                # 비동기 처리로 다른 작업이 실행될 수 있도록 함
                await asyncio.sleep(0.01)

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
