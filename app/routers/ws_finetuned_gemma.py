from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.ws_llm_streamer import WsLLMStreamer


router = APIRouter()


@router.websocket("/ws-llm-stream")
async def ws_llm_stream(websocket: WebSocket):
    await websocket.accept()
    print("FastAPI WebSocket 연결됨")
    
    try:
        while True:
            message = await websocket.receive_text()
            print(f"FastAPI에서 받은 메시지: {message}")
            
            # LLM 스트리밍 시작
            print("LLM 스트리밍 시작...")
            try:
                async for chunk in WsLLMStreamer.stream_answer(message):
                    print(f"토큰 전송: {chunk}")
                    await websocket.send_text(chunk)
                
                # 스트리밍 완료 신호
                await websocket.send_text("\n--- 스트리밍 완료 ---\n")
                print("LLM 스트리밍 완료")
                
            except Exception as e:
                print(f"LLM 스트리밍 오류: {e}")
                await websocket.send_text(f"LLM 오류: {str(e)}")
            
    except WebSocketDisconnect:
        print("FastAPI WebSocket 연결 종료")
    except Exception as e:
        print(f"FastAPI WebSocket 오류: {e}")
        try:
            await websocket.send_text(f"오류: {str(e)}")
        except:
            pass
        finally:
            await websocket.close()

