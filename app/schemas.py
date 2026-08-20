
from typing import Literal
from pydantic import BaseModel, Field

class QueryClassification(BaseModel):
    query_type: Literal["specific_ref", "conceptual", "multi_hop"] = Field(
        description="The category this query belongs to"
    )
    reasoning: str = Field(description="One sentence explaining why")