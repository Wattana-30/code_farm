from datetime import datetime
from pydantic import BaseModel

# schemas
from schemas.schemas import PeeweeGetterDict
from schemas import farm_schemas


class GreenHouseBase(BaseModel):
    temp: float
    humid: float
    ec: float
    co2: float
    created_at: datetime


class GreenHouseCreate(GreenHouseBase):
    pass


class GreenHouse(GreenHouseBase):
    id: int
    # farm: farm_schemas.Farm
    farm_id: int
    created_at: datetime

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict
