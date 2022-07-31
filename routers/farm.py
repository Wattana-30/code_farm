from typing import List
from fastapi import Depends, APIRouter

# databases
from databases.get_db import get_db

# schemas
from schemas import farm_schemas

# crud
from crud import farm_crud as crud


router_farms = APIRouter(
    prefix="/farms",
    tags=["farms"]
)

'''
    GET     Read data
    POST    Create data
    PUT     Update data
    DELETE  Delete data
'''

# สร้าง farm นะ
@router_farms.post("/", response_model=farm_schemas.Farm, dependencies=[Depends(get_db)])
def create_farm(item: farm_schemas.FarmCreate):
    return crud.create_farm(item)


# อ่านข้อมูล
@router_farms.get("/", response_model=List[farm_schemas.Farm], dependencies=[Depends(get_db)])
def read_farms(skip: int = 0, limit: int = 100):
    return crud.read_farms(skip, limit)


@router_farms.get("/{id}", response_model=farm_schemas.Farm, dependencies=[Depends(get_db)])
def read_farm(id: int):
    return crud.read_farm(id)


# อัปเดต
@router_farms.put("/", response_model=farm_schemas.FarmBase, dependencies=[Depends(get_db)])
def update_farm(farm_id: int, item: farm_schemas.FarmBase):
    return crud.update_farm(item, farm_id)


# ลบข้อมูล
@router_farms.delete("/{id}", dependencies=[Depends(get_db)])
def delete_farm(id: int):
    return crud.delete_farm(id)
