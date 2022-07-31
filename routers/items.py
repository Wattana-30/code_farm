from typing import List

from fastapi import Depends, APIRouter

# crud
from crud import crud

# databases
from databases.get_db import get_db

# schemas
from schemas import schemas

router_items = APIRouter(
    prefix="/items",
    tags=["items"]
)

@router_items.get("/", response_model=List[schemas.Item], dependencies=[Depends(get_db)])
def read_items(skip: int = 0, limit: int = 100):
    items = crud.get_items(skip=skip, limit=limit)
    return items

