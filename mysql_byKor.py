from cgitb import grey
import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import os
from PIL import Image
import io
import json
from uuid import uuid4
import statistics as st
from ast import Num
import time
# crud
from datetime import datetime

from crud import plant_features_crud
from crud import farm_crud
import mysql.connector
# schemas
from schemas import plant_features_schemas
from schemas import farm_schemas
import process as utils

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="plant_monitoring_robot"
)

mycursor = mydb.cursor()



temp = 21.5
humid = 45.8
ec = 1410.85
co2 = 1285.5



def insert_plant_features():
    import glob
    from uuid import uuid4
    # assign directory
    directory = 'images/NDVI'
    directory_1 = 'images/RGB'
    directory_2 = 'images/NIR'
    # iterate over files in
    # that directory
    for x in os.listdir(directory):
        ndvi_list = os.path.join(directory, x)
        mean_ndvi, std_ndvi = utils.findNdviValue(ndvi_list)
        # checking if it is a file
        if os.path.isfile(ndvi_list):
            head_tail = os.path.splitext(ndvi_list)
            txt = head_tail[0]
            split_txt = txt.split("/")
            txt_2 = split_txt[1].split("\\")
            retext = txt_2[1]
            newtext = retext.split("_")
            farm_loc = newtext[0]+"_"+newtext[1]
            #print("Farm: "+farm_loc)
            dt_txt = newtext[2].split(" ")
            # datetime >>> created_at
            date_time = dt_txt[0]+":"+dt_txt[1]+"_"+newtext[3]+"_"+newtext[4]
            date_time_split = date_time.split(":")
            #print(date_time_split)
            #print("Time: "+date_time_split[0]+" : "+date_time_split[1])
            # checking if it is a file
        for i in os.listdir(directory_1):
            rgb_list = os.path.join(directory_1, i)
            if i == x:
                try:
                    rgb_path = rgb_list
                    leaf_area_index = utils.findLeafArea(rgb_path)
                    mean_rgb, std_rgb = utils.findRGBValue(rgb_path)
                except:
                    os.remove(ndvi_list)
                    break
        for j in os.listdir(directory_2):
            nir_list = os.path.join(directory_2, j)
            if j == x:
                nir_path = nir_list
                break

        sql = f'''
        INSERT INTO greenhouse_env (farm_id,	temp,	humid,	ec,	co2,	created_at)
        VALUES (1, {temp}, {humid}, {ec}, {co2}, '{str(date_time())}')
        '''

        mycursor.execute(sql)
        mydb.commit()
        print(mycursor.rowcount, "record inserted.")

        sql = f'''
        INSERT INTO plant_features (green_id,	farm_id,	plant_loc,	rgb_path,	noir_path,	ndvi_path,	leaf_area_index,	created_at)
        VALUES ({1}, {1}, '{farm_loc}', '{rgb_path}', '{nir_path}', '{ndvi_list}', {leaf_area_index}, '{str(date_time)}')
        '''
        mycursor.execute(sql)
        mydb.commit()
        print(mycursor.rowcount, "record inserted.")



insert_plant_features()