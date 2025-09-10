from fastapi import APIRouter, HTTPException
from app.services.skin_trouble import load_model, preprocess, predict
from app.services.parsing_patientdata import PatientData
from app.schema.skin_trouble_schema import SkinTroubleUUIDLIST
from dotenv import load_dotenv
import os
load_dotenv()
ORTHANC_URL = os.getenv("ORTHANC_URL")
router = APIRouter()
model = load_model()
class_map = {
    0: "광선각화증",
    1: "기저세포암",
    2: "멜라닌세포모반",
    3: "보웬병",
    4: "비립종",
    5: "사마귀",
    6: "악성흑생종",
    7: "지루각화증",
    8: "편평세포암",
    9: "표피낭종",
    10: "피부섬유종",
    11: "피지샘증식증",
    12: "혈관종",
    13: "화농육아종",
    14: "흑색점"
}

@router.post("/prediction")
async def make_prediction(instanceUUID:SkinTroubleUUIDLIST):
    """피부 질환 예측"""
    try:
        parsing_service = PatientData(ORTHANC_URL)
        instances_images = parsing_service.parsing_target_dicom_image([instanceUUID.skin_trouble_uuid_list])
        processed_data = preprocess(instances_images)
        probs_dict, pred_label = predict(model, processed_data)
        return {"prediction": probs_dict, "label" : pred_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        