from pydantic import BaseModel
from typing import List, Dict

class AppendicitisUUIDLIST(BaseModel):
    AppendicitisUuidList: List[str]

class AppendicitisDescription(BaseModel):
    AppendicitisDescription : Dict
    