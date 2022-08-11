from datetime import datetime
from pydantic import BaseModel

# schemas
from schemas.schemas import PeeweeGetterDict


class PlantFeaturesBase(BaseModel):
    plant_loc: str
    rgb_path: str
    noir_path: str
    ndvi_path: str
    leaf_area_index: float
    created_at: datetime


class PlantFeaturesCreate(PlantFeaturesBase):
    pass


class PlantFeatures(PlantFeaturesBase):
    id: int
    # qrcode_id: int
    created_at: datetime

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict
