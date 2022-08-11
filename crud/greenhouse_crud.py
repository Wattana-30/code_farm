from datetime import datetime
from fastapi import HTTPException

# databases
from databases import models

# schemas
from schemas import greenhouse_schemas as schemas


def create_greenhouse(farm_id: int, item: schemas.GreenHouseCreate):
    db_greenhouse = models.GreenhouseEnv(**item.dict(), farm_id=farm_id)
    try: db_greenhouse.save()
    except: raise HTTPException(status_code=400, detail=f"ไม่มี farm_id: {farm_id}")
    return db_greenhouse


def read_greenhouses(skip: int = 0, limit: int = 100):
    return list(models.GreenhouseEnv.select().offset(skip).limit(limit))


def read_greenhouse(id: int):
    return models.GreenhouseEnv.filter(models.GreenhouseEnv.id == id).first()


def update_greenhouse(green_id: int, item: schemas.GreenHouseBase):
    models.GreenhouseEnv.update(**item.dict()).where(models.GreenhouseEnv.id == green_id).execute()
    return item


def delete_greenhouse(id: int):
    models.GreenhouseEnv.delete().where(models.GreenhouseEnv.id == id).execute()
    return {'message': f'delete id is {id} successful.'}


def pack_and_insert_greenhouse(msg: str):
    msgList = msg.split(" ")

    farm_id = msgList[0]
    co2 = msgList[1]
    ec = msgList[2]
    rh = msgList[3]
    temp = msgList[4]

    item = schemas.GreenHouseCreate(
        temp=temp, humid=rh, ec=ec, co2=co2, created_at=datetime.now()
    )

    create_greenhouse(farm_id, item)
    