from datetime import datetime
from pydantic import BaseModel

# schemas
from schemas.schemas import PeeweeGetterDict
from schemas import farm_schemas


class PlantQrcodeBase(BaseModel):
    tree_index: int
    qr_code: str
    


class PlantQrcodeCreate(PlantQrcodeBase):
    pass


class PlantQrcode(PlantQrcodeBase):
    id: int
    # farm: farm_schemas.Farm
    farm_id: int
    created_at: datetime

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict
