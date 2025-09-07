from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import finetuned_gemma, segmentation_model, image_meta, ws_finetuned_gemma, ws_nlp_streamer
from app.services.ws_llm_streamer import WsLLMStreamer
from app.services.ws_nlp_streamer import WsNLPStreamer
import uvicorn

app = FastAPI()

# 애플리케이션 시작 시 모든 모델 사전 로딩
@app.on_event("startup")
async def startup_event():
    print("🚀 FastAPI 애플리케이션 시작 중...")
    try:
        # LLM 모델 사전 로딩
        WsLLMStreamer.preload_model()
        print("✅ LLM 모델 로딩 완료!")
        
        # NLP 모델 사전 로딩
        WsNLPStreamer.preload_model()
        print("✅ NLP 모델 로딩 완료!")
        
        print("✅ 모든 모델 로딩 완료!")
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        # 모델 로딩 실패해도 애플리케이션은 계속 실행

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # 모든 출처 허용
    allow_credentials=True, # 인증정보(쿠키 등)포함 가능
    allow_methods=["*"],    # 모든 HTTP 메서드(GET, POST 등) 허용
    allow_headers=["*"],    # 모든 헤더 허용
)

# 라우터 등록
app.include_router(finetuned_gemma.router, prefix="/chat-bot", tags=["chat-bot"])
app.include_router(segmentation_model.router, prefix="/image", tags=["image-segmentation"])
app.include_router(image_meta.router, prefix='/image-meta', tags = ["image-meta-diagnosis"])
app.include_router(ws_finetuned_gemma.router, prefix="/chat-bot-ws", tags=["chat-bot-ws"])
app.include_router(ws_nlp_streamer.router, prefix="/image-meta-ws", tags=["image-meta-diagnosis-ws"])

# FAST 실행명령어 자동 실행
if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)  # reload=True : 코드 변경 시 서버 자동 재시작
    