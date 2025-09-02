from fastapi import APIRouter, HTTPException, WebSocket
from app.services.finetuned_llm import FineTunedLLMModel
from app.schema.chat_schema import ChatRequest, ChatResponse
import re

router = APIRouter()
gemma3 = FineTunedLLMModel()

@router.post("/ask-llm", response_model=ChatResponse)
async def ask_question(question:ChatRequest):
    """
    FineTuning된 LLM Gemma3-1b에게 질문합니다.
    """
    try:
        # Generate Texts 생성
        response = gemma3.generate_answer(question.message)
        response = re.sub(r"[^가-힣0-9\s,]", "", response).replace("\n","").rstrip()
        return ChatResponse(response=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.websocket("/ws-llm")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    while True:
        message = await websocket.receive_text()
        # LLM 처리
        response = gemma3.generate_answer(message)
        response = re.sub(r"[^가-힣0-9\s,]", "", response).replace("\n","").rstrip()
        await websocket.send_text(response)