from fastapi import HTTPException

# databases
from databases import models

# schemas
from schemas import plant_features_schemas as schemas


def create_plant_features(qrcode_id: int, item: schemas. PlantFeaturesBase):
    db_plant_features = models. PlantFeatures(**item.dict(), qrcode_id=qrcode_id)
    try: db_plant_features.save()
    except: raise HTTPException(status_code=400, detail=f"ไม่มี qrcode_id: {qrcode_id} or มีข้อมูล rgb_path, noir_path แล้ว")
    return db_plant_features


def read_plant_featuress(skip: int = 0, limit: int = 100):
    return list(models.PlantFeatures.select().offset(skip).limit(limit))


def read_plant_features(id: int):
    return models.PlantFeatures.filter(models.PlantFeatures.id == id).first()


def update_plant_features(plant_features_id: int, item: schemas. PlantFeaturesBase):
    models.PlantFeatures.update(**item.dict()).where(models.PlantFeatures.id == plant_features_id).execute()
    return item


def delete_plant_features(id: int):
    models.PlantFeatures.delete().where(models.PlantFeatures.id == id).execute()
    return {'message': f'delete id is {id} successful.'}
