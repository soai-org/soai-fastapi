from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.ws_nlp_streamer import WsNLPStreamer
from app.services.parsing_patientdata import PatientData
import json
import torch
import asyncio
import os

router = APIRouter()

ORTHANC_URL = os.getenv("ORTHANC_URL")

mapping = {'선천성유문협착증':'Pyloric Stenosis','공기액체음영':'air-fluid level',
           '기복증':'Abdominal distension','변비':'Constipation','정상':'Normal'}
parsing_patient = PatientData(ORTHANC_URL)

@router.websocket("/ws-nlp-stream")
async def ws_nlp_stream(websocket: WebSocket):
    await websocket.accept()
    print("FastAPI WebSocket 연결됨")
    
    try:
        while True:
            instance_uuid = await websocket.receive_text()
            print(f"FastAPI에서 받은 instance_uuid: {instance_uuid}")
            description = await websocket.receive_text()
            print(f"FastAPI에서 받은 description: {description}")
            
            instances_image = parsing_patient.parsing_target_dicom_image([instance_uuid])
            instances_image = instances_image / 255.0
            image_tensor = torch.from_numpy(instances_image).permute(0,3,1,2).to('cpu').float()
            meta_label = mapping[description]
            
            print("NLP 캡션 스트리밍 시작...")
            try:
                async for chunk in WsNLPStreamer.stream_caption_generation(image_tensor, meta_label):
                    print(f"토큰 전송: {chunk}")
                    await websocket.send_text(chunk)
                
                # 스트리밍 완료 신호
                await websocket.send_text("\n--- 캡션 생성 완료 ---\n")
                print("NLP 캡션 스트리밍 완료")
                
            except Exception as e:
                print(f"NLP 캡션 스트리밍 오류: {e}")
                await websocket.send_text(f"NLP 오류: {str(e)}")
            
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