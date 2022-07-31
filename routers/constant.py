from typing import List
from fastapi import Depends, APIRouter

# databases
from databases.get_db import get_db

# schemas
from schemas import constant_schemas as schemas

# crud
from crud import constant_crud as crud


router_constant = APIRouter(
    prefix="/constant",
    tags=["constant"]
)

'''
    GET     Read data
    POST    Create data
    PUT     Update data
    DELETE  Delete data
'''

# สร้าง constant นะ
@router_constant.post("/", response_model=schemas.Constant, dependencies=[Depends(get_db)])
def create_constant(farm_id: int, item: schemas.ConstantBase):
    return crud.create_constant(farm_id, item)


# อ่านข้อมูล  
@router_constant.get("/", response_model=List[schemas.Constant], dependencies=[Depends(get_db)])
def read_constants(skip: int = 0, limit: int = 100):
    return crud.read_constants(skip, limit)



@router_constant.get("/{id}", response_model=schemas.Constant, dependencies=[Depends(get_db)])
def read_constant(id: int):
    return crud.read_constant(id)


# อัปเดต
@router_constant.put("/", response_model=schemas.ConstantBase, dependencies=[Depends(get_db)])
def update_constant(constant_id: int, item: schemas.ConstantBase):
    return crud.update_constant(constant_id, item)


# ลบข้อมูล
@router_constant.delete("/{id}", dependencies=[Depends(get_db)])
def delete_constant(id: int):
    return crud.delete_constant(id)
