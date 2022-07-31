from datetime import datetime
from pydantic import BaseModel

# schemas
from schemas.schemas import PeeweeGetterDict


class FarmBase(BaseModel):
    farm_name: str


class FarmCreate(FarmBase):
    pass


class Farm(FarmCreate):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict


