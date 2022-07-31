from fastapi import HTTPException

# databases
from databases import models

# schemas
from schemas import constant_schemas as schemas


def create_constant(farm_id: int, item: schemas.ConstantBase):
    db_tree = models.Constant(**item.dict(), farm_id=farm_id)
    try: db_tree.save()
    except: raise HTTPException(status_code=400, detail=f"ไม่มี farm_id: {farm_id} or มีข้อมูล constant แล้ว")
    return db_tree


def read_constants(skip: int = 0, limit: int = 100):
    return list(models.Constant.select().offset(skip).limit(limit))


def read_constant(id: int):
    return models.Constant.filter(models.Constant.id == id).first()


def update_constant(constant_id: int, item: schemas.ConstantBase):
    models.Constant.update(**item.dict()).where(models.Constant.id == constant_id).execute()
    return item


def delete_constant(id: int):
    models.Constant.delete().where(models.Constant.id == id).execute()
    return {'message': f'delete id is {id} successful.'}
