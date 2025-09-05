from pydantic import BaseModel
from typing import List, Dict, Any

class AppendicitisUUIDLIST(BaseModel):
    AppendicitisUuidList: List[str]
    
class AppendicitisDescription(BaseModel):
    AppendicitisDescription: Dict[str, Any]
