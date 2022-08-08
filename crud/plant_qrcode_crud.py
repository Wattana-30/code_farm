from fastapi import HTTPException

# databases
from databases import models

# schemas
from schemas import plant_qrcode_schemas as schemas


def create_plant_qrcode(farm_id: int, item: schemas.PlantQrcodeCreate):
    db_plant_qrcode = models.PlantQrcode(**item.dict(), farm_id=farm_id)
    try: db_plant_qrcode.save()
    except: raise HTTPException(status_code=400, detail=f"ไม่มี farm_id: {farm_id} or qr code ซ้ำกัน")
    return db_plant_qrcode


def read_plant_qrcodes(skip: int = 0, limit: int = 100):
    return list(models.PlantQrcode.select().offset(skip).limit(limit))


def read_plant_qrcode(id: int):
    return models.PlantQrcode.filter(models.PlantQrcode.id == id).first()


def update_plant_qrcode(plant_qrcode_id: int, item: schemas.PlantQrcodeBase):
    models.PlantQrcode.update(**item.dict()).where(models.PlantQrcode.id == plant_qrcode_id).execute()
    return item


def delete_plant_qrcode(id: int):
    models.PlantQrcode.delete().where(models.PlantQrcode.id == id).execute()
    return {'message': f'delete id is {id} successful.'}


