from fastapi import APIRouter, HTTPException
from app.services.Multiview import MultiViewService
from app.services.parsing_patientdata import PatientData
from app.schema.appendicitis_schema import AppendicitisUUIDLIST, AppendicitisDescription
from dotenv import load_dotenv
import os
load_dotenv()
ORTHANC_URL = os.getenv("ORTHANC_URL")
router = APIRouter()
business_service = MultiViewService()
appendicitis_model = business_service._load_model()

@router.post("/diagnosis", response_model=AppendicitisDescription)
async def ask_question(instanceUUID:AppendicitisUUIDLIST):
    """충수염 판단 모델"""
    try:
        parsing_service = PatientData(ORTHANC_URL)
        instances_images = parsing_service.parsing_target_dicom_images(instanceUUID.appendicitisUuidList)
        images, mask, _ = business_service.create_multiview_batch(instances_images)
        results = business_service.predict_multiview_tensor(images, mask)
        results = {"appendcitis_probability" : results['results']['patient_0']['appendicitis_probability'],
                   "concept_scores" : results['results']['patient_0']['concept_scores'],
                   "num_views" : results['results']['patient_0']['num_views']}
        return AppendicitisDescription(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))