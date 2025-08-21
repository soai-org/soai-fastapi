from pydantic import BaseModel

class patient_study(BaseModel):
    study : str

class captioning_message(BaseModel):
    transcript: str