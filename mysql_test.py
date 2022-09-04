import os
import mysql.connector
from datetime import datetime
import random


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="plant_monitoring_robot"
)

mycursor = mydb.cursor()


def insert_env():
  temp = random.randint(20, 70)
  humid = random.randint(100, 520)
  ec = random.randint(100, 200)
  co2 = random.randint(1000, 2000)

  sql = f'''
    INSERT INTO greenhouse_env (farm_id,	temp,	humid,	ec,	co2,	created_at)
    VALUES (1, {temp}, {humid}, {ec}, {co2}, '{str(datetime.now())}')
  '''

  mycursor.execute(sql)
  mydb.commit()
  print(mycursor.rowcount, "record inserted.")


def insert_plant_features():
  import glob
  from uuid import uuid4

  image_list = []
  for filename in glob.glob('images/NDVI/*.jpg'):
    path = filename.replace("/", "\\").split("\\")[-1]
    image_list.append(path)

  plant_loc = ['Point_1', 'Point_2', 'Point_3', 'Point_4', 'Point_5', 'Point_6']

  for id in range(1, 100):
    for pl in plant_loc:
      path = os.path.join(os.getcwd(),  str(uuid4()).split('-')[-1], random.choice(image_list))
      path = path.replace('\\', '/')

      leaf_area_index = random.randint(1000, 2000)

      sql = f'''
        INSERT INTO plant_features (green_id,	farm_id,	plant_loc,	rgb_path,	noir_path,	ndvi_path,	leaf_area_index,	created_at)
        VALUES ({id}, 1, '{pl}', '{path}', '{path}', '{path}', {leaf_area_index}, '{str(datetime.now())}')
      '''

      mycursor.execute(sql)
      mydb.commit()
      print(mycursor.rowcount, "record inserted.")


if __name__ == '__main__':
  # for i in range(0, 100):
  #   insert_env()

  insert_plant_features()