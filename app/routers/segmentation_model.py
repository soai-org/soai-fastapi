from fastapi import APIRouter, HTTPException
from app.services.segmentation import load_model, predict
from app.schema.segmentation_request_schema import segmentation_request
from app.services.parsing_patientdata import PatientData
import torch
import io
from PIL import Image
import numpy as np
from fastapi.responses import StreamingResponse
import os

ORTHANC_URL = os.getenv("ORTHANC_URL")
parse = PatientData(ORTHANC_URL)
router = APIRouter()
model = load_model()

@router.post("/segmentation")
async def segmentation(request:segmentation_request):
    try:
        # 1️⃣ 원본 DICOM → torch 변환
        instances_image = parse.parsing_target_dicom_image([request.instanceUUID])  # (N,H,W,3)
        instances_image = instances_image / 255.0  # 0~1 normalize
        tensor = torch.from_numpy(instances_image).permute(0,3,1,2).to('cpu').float()

        # 2️⃣ 세그멘테이션
        segmented_image = predict(model, tensor)  # (H, W)
        segmented_image_np = (segmented_image > 0.5).astype(np.uint8)  # binary mask (0/1)

        # 3️⃣ overlay 색상 정의 (예: 빨간색)
        overlay_color = np.array([255, 0, 0], dtype=np.uint8)  # 빨강
        alpha = 0.4  # 투명도

        # 4️⃣ 원본 이미지 준비
        base_img = (instances_image[0] * 255).astype(np.uint8)  # 첫 slice
        base_img = np.array(base_img, dtype=np.uint8)

        # 5️⃣ mask 적용
        mask_3ch = np.stack([segmented_image_np]*3, axis=-1)  # (H,W,3)
        overlay = np.where(mask_3ch==1, 
                           (base_img*(1-alpha) + overlay_color*alpha).astype(np.uint8),
                           base_img)

        # 6️⃣ PIL 이미지 변환
        img = Image.fromarray(overlay)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
