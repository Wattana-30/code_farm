from typing import List
from fastapi import Depends, APIRouter

# databases
from databases.get_db import get_db

# schemas
from schemas import greenhouse_schemas as schemas

# crud
from crud import greenhouse_crud as crud


router_greenhouses = APIRouter(
    prefix="/greenhouses",
    tags=["greenhouses"]
)

'''
    GET     Read data
    POST    Create data
    PUT     Update data
    DELETE  Delete data
'''

# สร้าง greenhouse นะ
@router_greenhouses.post("/", response_model=schemas.GreenHouse, dependencies=[Depends(get_db)])
def create_greenhouse(farm_id: int, item: schemas.GreenHouseCreate):
    return crud.create_greenhouse(farm_id, item)


# อ่านข้อมูล  
@router_greenhouses.get("/", response_model=List[schemas.GreenHouse], dependencies=[Depends(get_db)])
def read_greenhouses(skip: int = 0, limit: int = 100):
    return crud.read_greenhouses(skip, limit)


@router_greenhouses.get("/{id}", response_model=schemas.GreenHouse, dependencies=[Depends(get_db)])
def read_greenhouse(id: int):
    return crud.read_greenhouse(id)


# อัปเดต
@router_greenhouses.put("/", response_model=schemas.GreenHouseBase, dependencies=[Depends(get_db)])
def update_greenhouse(green_id: int, item: schemas.GreenHouseBase):
    return crud.update_greenhouse(green_id, item)


# ลบข้อมูล
@router_greenhouses.delete("/{id}", dependencies=[Depends(get_db)])
def delete_greenhouse(id: int):
    return crud.delete_greenhouse(id)
