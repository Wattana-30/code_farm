from typing import List
from fastapi import Depends, APIRouter

# databases
from databases.get_db import get_db

# schemas
from schemas import plant_qrcode_schemas as schemas

# crud
from crud import plant_qrcode_crud as crud


router_plant_qrcode = APIRouter(
    prefix="/plant_qrcode",
    tags=["plant_qrcode"]
)

'''
    GET     Read data
    POST    Create data
    PUT     Update data
    DELETE  Delete data
'''

# สร้าง tree นะ
@router_plant_qrcode.post("/", response_model=schemas.PlantQrcode, dependencies=[Depends(get_db)])
def create_plant_qrcode(farm_id: int, item: schemas.PlantQrcodeCreate):
    return crud.create_plant_qrcode(farm_id, item)


# อ่านข้อมูล  
@router_plant_qrcode.get("/", response_model=List[schemas.PlantQrcode], dependencies=[Depends(get_db)])
def read_plant_qrcodes(skip: int = 0, limit: int = 100):
    return crud.read_plant_qrcodes(skip, limit)


@router_plant_qrcode.get("/{id}", response_model=schemas.PlantQrcode, dependencies=[Depends(get_db)])
def read_plant_qrcode(id: int):
    return crud.read_plant_qrcode(id)


# อัปเดต
@router_plant_qrcode.put("/", response_model=schemas.PlantQrcode, dependencies=[Depends(get_db)])
def update_plant_qrcode(plant_qrcode_id: int, item: schemas.PlantQrcode):
    return crud.update_plant_qrcode(plant_qrcode_id, item)


# ลบข้อมูล
@router_plant_qrcode.delete("/{id}", dependencies=[Depends(get_db)])
def delete_plant_qrcode(id: int):
    return crud.delete_plant_qrcode(id)
