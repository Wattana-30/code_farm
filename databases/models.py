from datetime import datetime
import peewee

# databases
from databases.database import db


class User(peewee.Model):
    email = peewee.CharField(unique=True, index=True)
    hashed_password = peewee.CharField()
    is_active = peewee.BooleanField(default=True)

    class Meta:
        database = db


class Item(peewee.Model):
    title = peewee.CharField(index=True)
    description = peewee.CharField(index=True)
    owner = peewee.ForeignKeyField(User, backref="items")

    class Meta:
        database = db


class Farm(peewee.Model):
    farm_name = peewee.CharField(max_length=255, unique=True)
    created_at = peewee.DateTimeField(default=datetime.now())

    class Meta:
        db_table = 'farm'
        database = db


class GreenhouseEnv(peewee.Model):
    farm = peewee.ForeignKeyField(Farm)
    temp = peewee.DoubleField()
    humid = peewee.DoubleField()
    ec = peewee.DoubleField()
    co2 = peewee.DoubleField()
    created_at = peewee.DateTimeField(default=datetime.now())

    class Meta:
        db_table = 'greenhouse_env'
        database = db


class PlantQrcode(peewee.Model):
    farm = peewee.ForeignKeyField(Farm)
    tree_index = peewee.IntegerField()
    qr_code = peewee.CharField(max_length=255, unique=True)
    created_at = peewee.DateTimeField(default=datetime.now())

    class Meta:
        db_table = 'plant_qrcode'
        database = db


class Constant(peewee.Model):
    farm = peewee.ForeignKeyField(Farm, unique=True)
    height = peewee.DoubleField()
    created_at = peewee.DateTimeField(default=datetime.now())

    class Meta:
        db_table = 'constant'
        database = db


class PlantFeatures(peewee.Model):
    # qrcode = peewee.ForeignKeyField(PlantQrcode)
    green_id = peewee.ForeignKeyField(GreenhouseEnv)
    farm = peewee.ForeignKeyField(Farm)
    plant_loc = peewee.CharField(max_length=255)
    rgb_path = peewee.CharField(max_length=255, unique=True)
    mean_rgb = peewee.DoubleField()
    std_rgb = peewee.DoubleField()
    noir_path = peewee.CharField(max_length=255, unique=True)
    ndvi_path = peewee.CharField(max_length=255, unique=True)
    mean_ndvi = peewee.DoubleField()
    std_ndvi = peewee.DoubleField()
    leaf_area_index = peewee.DoubleField()
    created_at = peewee.DateTimeField(default=datetime.now())

    class Meta:
        db_table = 'plant_features'
        database = db




