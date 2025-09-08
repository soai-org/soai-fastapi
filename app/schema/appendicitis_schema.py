from pydantic import BaseModel
from typing import List, Dict, Any

class AppendicitisUUIDLIST(BaseModel):
    appendicitisUuidList: List[str]
    
class AppendicitisDescription(BaseModel):
    appendcitis_probability: float
    concept_scores: Dict[str, float]
    num_views: int
