import time
from typing import List

from fastapi import Depends, APIRouter, HTTPException

# crud
from crud import crud

# databases
from databases.get_db import get_db

# schemas
from schemas import schemas


router_users = APIRouter(
    prefix="/users",
    tags=["users"]
)

sleep_time = 10

@router_users.post("/", response_model=schemas.User, dependencies=[Depends(get_db)])
def create_user(user: schemas.UserCreate):
    db_user = crud.get_user_by_email(email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(user=user)


@router_users.get("/", response_model=List[schemas.User], dependencies=[Depends(get_db)])
def read_users(skip: int = 0, limit: int = 100):
    users = crud.get_users(skip=skip, limit=limit)
    return users


@router_users.get(
    "/{user_id}", response_model=schemas.User, dependencies=[Depends(get_db)]
)
def read_user(user_id: int):
    db_user = crud.get_user(user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router_users.post(
    "/{user_id}/items/",
    response_model=schemas.Item,
    dependencies=[Depends(get_db)],
)
def create_item_for_user(user_id: int, item: schemas.ItemCreate):
    return crud.create_user_item(item=item, user_id=user_id)


@router_users.get(
    "/slowusers/", response_model=List[schemas.User], dependencies=[Depends(get_db)]
)
def read_slow_users(skip: int = 0, limit: int = 100):
    global sleep_time
    sleep_time = max(0, sleep_time - 1)
    time.sleep(sleep_time)  # Fake long processing request
    users = crud.get_users(skip=skip, limit=limit)
    return users
