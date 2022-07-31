from typing import List
from fastapi import Depends, APIRouter

# databases
from databases.get_db import get_db

# schemas
from schemas import plant_features_schemas as schemas

# crud
from crud import plant_features_crud as crud


router_plant_features = APIRouter(
    prefix="/plant_features",
    tags=["plant_features"]
)

'''
    GET     Read data
    POST    Create data
    PUT     Update data
    DELETE  Delete data
'''

# สร้าง plant_featuresนะ


@router_plant_features.post("/", response_model=schemas.PlantFeatures, dependencies=[Depends(get_db)])
def create_ndvi(qrcode_id: int, item: schemas.PlantFeaturesBase):
    return crud.create_plant_features(qrcode_id, item)


# อ่านข้อมูล
@router_plant_features.get("/", response_model=List[schemas.PlantFeatures], dependencies=[Depends(get_db)])
def read_plant_featuress(skip: int = 0, limit: int = 100):
    return crud.read_plant_featuress(skip, limit)


@router_plant_features.get("/{id}", response_model=schemas.PlantFeatures, dependencies=[Depends(get_db)])
def read_plant_features(id: int):
    return crud.read_plant_features(id)


# อัปเดต
@router_plant_features.put("/", response_model=schemas.PlantFeaturesBase, dependencies=[Depends(get_db)])
def update_plant_features(plant_features_id: int, item: schemas.PlantFeaturesBase):
    return crud.update_plant_features(plant_features_id, item)


# ลบข้อมูล
@router_plant_features.delete("/{id}", dependencies=[Depends(get_db)])
def delete_plant_features(id: int):
    return crud.delete_plant_features(id)
