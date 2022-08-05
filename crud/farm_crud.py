from fastapi import HTTPException

# models
from databases import models

# schemas
from schemas import farm_schemas


def create_farm(item: farm_schemas.Farm):
    db_farm = models.Farm(**item.dict())
    try:
        db_farm.save()
    except:
        raise HTTPException(
            status_code=400, detail=f"{item.farm_name} มีชื่อนี้แล้ว")
    return db_farm


def read_farms(skip: int = 0, limit: int = 100):
    return list(models.Farm.select().offset(skip).limit(limit))


def read_farm(id: int):
    return models.Farm.filter(models.Farm.id == id).first()


def update_farm(item: farm_schemas.FarmBase, farm_id: int):
    models.Farm.update(
        **item.dict()).where(models.Farm.id == farm_id).execute()
    return item


def delete_farm(id: int):
    models.Farm.delete().where(models.Farm.id == id).execute()
    return {'message': f'delete farm_id is {id} successful.'}


def pack_and_insert_Farm(msg: str):
    msgList = msg.split(" ")

    farm_name = msgList[0]

    item = farm_schemas.FarmCreate(
        farm_name=farm_name,
    )

    create_farm(farm_name, item)
