from typing import List, Optional
from pydantic import BaseModel

class UpdateDocumentsRequest(BaseModel):
    file_paths: Optional[List[str]] = None

class UpdateDocumentsResponse(BaseModel):
    message: str 