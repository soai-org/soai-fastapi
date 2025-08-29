from fastapi import APIRouter, HTTPException
from app.services.vis_nlp import ViTFeatureExtractor, BertFeatureExtractor, CrossAttention, VISNLPEXTRACTOR, CaptionDecoder, CaptionGenerator, load_Generator, generate_caption
from app.schema.image_meta_schema import patient, captioning_message
from app.services.parsing_patientdata import PatientData
import torch
from transformers import AutoTokenizer
# router 설정
router = APIRouter()
import os

# Model Config
ORTHANC_URL = "http://127.0.0.1:8042"
IMG_WIDTH = 512
IMG_DIM_HEIGHT = 512
PATCH_SIZE = 16
EMBED_DIM = 256
NUM_HEADS = 4
DEPTH = 4
FF_DIM = 2048
NUM_LAYERS = 4

# Meta Config
mapping = {'선천성유문협착증':'Pyloric Stenosis','공기액체음영':'air-fluid level',
           '기복증':'Abdominal distension','변비':'Constipation','정상':'Normal'}
parsing_patient = PatientData(ORTHANC_URL)

# Model 불러오기
model = CaptionGenerator(
    VISNLPEXTRACTOR(IMG_DIM_HEIGHT, IMG_WIDTH, PATCH_SIZE, EMBED_DIM, NUM_HEADS, DEPTH),
    vocab_size=30522,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS
).to('cpu')

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/image_meta.pth')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
captioning_model = load_Generator(model,MODEL_PATH)

@router.post("/diagnosis", response_model=captioning_message)
async def make_diagnosis(Patient:patient):
    """
        (기술 구현 어려움: 환자의 이전 과거 메타 데이터 참고해서 분석 진단 출력) 
        현재 메타 데이터를(진단 결과)와 이미지를 분석해 결과를 영문으로 출력합니다.
    """
    try:
        instances_image = parsing_patient.parsing_target_dicom_image([Patient.instanceUUID])

        instances_image = instances_image / 255.0  # 0~1 normalize
        image_tensor = torch.from_numpy(instances_image).permute(0,3,1,2).to('cpu').float()
        meta_label = mapping[Patient.description]

        with torch.no_grad():
            caption = generate_caption(captioning_model, image_tensor, meta_label, tokenizer, device='cpu')
        return captioning_message(transcript = caption)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
