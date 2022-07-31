from pydantic import BaseModel

# schemas
from schemas.schemas import PeeweeGetterDict
from schemas import farm_schemas


class ConstantBase(BaseModel):
    height: float


class ConstantCreate(ConstantBase):
    pass


class Constant(ConstantBase):
    id: int
    # farm: farm_schemas.Farm
    farm_id: int

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict
