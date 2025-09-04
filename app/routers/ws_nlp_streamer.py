from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.ws_nlp_streamer import WsNLPStreamer
import json
import torch
import asyncio


router = APIRouter()


@router.websocket("/ws-nlp-stream")
async def ws_nlp_stream(websocket: WebSocket):
    await websocket.accept()
    print("NLP WebSocket 연결됨")
    
    # NLP 스트리머 인스턴스 생성
    nlp_streamer = WsNLPStreamer()
    
    try:
        while True:
            message = await websocket.receive_text()
            print(f"NLP WebSocket에서 받은 메시지: {message}")
            
            try:
                # JSON 메시지 파싱
                data = json.loads(message)
                tensor_data = data.get('image_tensor')  # 텐서 데이터
                meta_prompt = data.get('meta_prompt', '이 이미지를 설명해주세요')
                
                if not tensor_data:
                    await websocket.send_text("텐서 데이터가 없습니다.")
                    continue
                
                # 텐서 데이터 처리
                try:
                    # 텐서 데이터를 PyTorch 텐서로 변환
                    # Spring Boot에서 보낸 텐서 데이터 형식에 따라 조정 필요
                    if isinstance(tensor_data, list):
                        # 리스트 형태로 온 경우 텐서로 변환
                        image_tensor = torch.tensor(tensor_data, dtype=torch.float32)
                    else:
                        # 이미 텐서 형태인 경우
                        image_tensor = tensor_data
                    
                    # 배치 차원 추가 (필요한 경우)
                    if image_tensor.dim() == 3:
                        image_tensor = image_tensor.unsqueeze(0)  # [B, C, H, W] 형태
                    
                    print("텐서 데이터 처리 완료, 캡션 생성 시작...")
                    
                    # NLP 스트리밍 시작
                    async for chunk in WsNLPStreamer.stream_caption_generation(image_tensor, meta_prompt):
                        print(f"캡션 토큰 전송: {chunk}")
                        await websocket.send_text(chunk)
                    
                    print("NLP 스트리밍 완료")
                    
                except Exception as e:
                    print(f"텐서 처리 오류: {e}")
                    await websocket.send_text(f"텐서 처리 오류: {str(e)}")
                
            except json.JSONDecodeError:
                await websocket.send_text("잘못된 JSON 형식입니다.")
            except Exception as e:
                print(f"NLP 스트리밍 오류: {e}")
                await websocket.send_text(f"NLP 오류: {str(e)}")
            
    except WebSocketDisconnect:
        print("NLP WebSocket 연결 종료")
    except Exception as e:
        print(f"NLP WebSocket 오류: {e}")
        try:
            await websocket.send_text(f"오류: {str(e)}")
        except:
            pass
        finally:
            await websocket.close()


@router.websocket("/ws-nlp-simple")
async def ws_nlp_simple(websocket: WebSocket):
    """
    간단한 텍스트 기반 NLP 스트리밍 (이미지 없이)
    """
    await websocket.accept()
    print("간단한 NLP WebSocket 연결됨")
    
    try:
        while True:
            message = await websocket.receive_text()
            print(f"간단한 NLP에서 받은 메시지: {message}")
            
            try:
                # 간단한 텍스트 응답 스트리밍
                response_parts = [
                    "안녕하세요! ",
                    "NLP 모델이 ",
                    "실시간으로 ",
                    "응답을 ",
                    "생성하고 있습니다. ",
                    "이것은 ",
                    "스트리밍 ",
                    "테스트입니다."
                ]
                
                for part in response_parts:
                    await websocket.send_text(part)
                    await asyncio.sleep(0.1)  # 실시간 효과
                
                await websocket.send_text("\n--- 응답 완료 ---\n")
                
            except Exception as e:
                print(f"간단한 NLP 오류: {e}")
                await websocket.send_text(f"오류: {str(e)}")
            
    except WebSocketDisconnect:
        print("간단한 NLP WebSocket 연결 종료")
    except Exception as e:
        print(f"간단한 NLP WebSocket 오류: {e}")
        try:
            await websocket.close()
        except:
            pass
