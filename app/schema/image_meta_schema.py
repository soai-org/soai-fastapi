from pydantic import BaseModel

class patient(BaseModel):
    instanceUUID : str
    description : str

class captioning_message(BaseModel):
    transcript: str