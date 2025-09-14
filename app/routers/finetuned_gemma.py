from fastapi import APIRouter, HTTPException, WebSocket
from app.services.finetuned_llm import FineTunedLLMModel
from app.schema.chat_schema import ChatRequest, ChatResponse
import re
import asyncio
from fastapi.websockets import WebSocketDisconnect

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
    try:
        while True:
            message = await websocket.receive_text()
            
            # 기존 POST 로직과 동일한 LLM 처리
            response = gemma3.generate_answer(message)
            response = re.sub(r"[^가-힣0-9\s,]", "", response).replace("\n","").rstrip()
            
            # 실시간 스트리밍: 문자 단위로 전송
            for char in response:
                await websocket.send_text(char)
                await asyncio.sleep(0.05)  # 스트리밍 효과
            
            # 응답 완료 구분자
            await websocket.send_text("\n")
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"__error__: {str(e)}")
        except Exception:
            pass
        finally:
            await websocket.close()