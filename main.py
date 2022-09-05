from datetime import datetime
import os
from fastapi_mqtt import FastMQTT, MQTTConfig
from fastapi import FastAPI
import uvicorn
import time
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse



# routers
from routers import plant_features, plant_qrcode, farm, greenhouse, constant

# databases
from databases import models
from databases.database import db

# crud
from crud import greenhouse_crud as greenEnv

# utils
from utils import utils

db.connect()
db.create_tables([
    models.Farm, 
    models.GreenhouseEnv, 
    models.PlantFeatures, 
    models.Constant
])
db.close()

app = FastAPI(title='Python x MySQL', description='APIs for MySQL Apis', version='0.1.0')





# app.include_router(items.router_items)
# app.include_router(users.router_users)
app.include_router(farm.router_farms)
app.include_router(greenhouse.router_greenhouses)
#app.include_router(plant_qrcode.router_plant_qrcode)
app.include_router(plant_features.router_plant_features)
app.include_router(constant.router_constant)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


mqtt_config = MQTTConfig(
    # host = "broker.emqx.io",
    host = "localhost",
    port= 1883,
    keepalive = 60,
    username="",
    password=""
)

mqtt = FastMQTT(
    config=mqtt_config,
    client_id="fastAPI x mysql x mqtt",
)

mqtt.init_app(app)


@mqtt.on_connect()
def connect(client, flags, rc, properties):
    print("Connected: ", client, flags, rc, properties)
    mqtt.client.subscribe("plant/farm")
    mqtt.client.subscribe("plant/plc-to-server")


@mqtt.on_message()
async def message(client, topic, payload, qos, properties):
    # print("Received message: ", topic, payload.decode(), qos, properties)
    # print("Received message: ", payload.decode())
    print("Topic: ", topic)
    print(f"message: {payload.decode()}")

    if topic == 'plant/farm':
        id = greenEnv.pack_and_insert_greenhouse(payload.decode())
        client.publish("plant/green-env-id", str(id))
    elif topic == 'plant/plc-to-server':
        payload = payload.decode()
        payloadList = payload.split(" ")

        # farm_id, position, ready, green_env_id

        if payloadList[1] == "false": return

        if payloadList[2] == "ready":
            dt = datetime.now()
            time.sleep(1)
            utils.startEvent(mqtt, payloadList[0], payloadList[1], payloadList[3], dt)
            
            client.publish("plant/server-to-plc", "ready")
            

@mqtt.on_disconnect()
def disconnect(client, packet, exc=None):
    print("Disconnected")


@mqtt.on_subscribe()
def subscribe(client, mid, qos, properties):
    print("subscribed", client, mid, qos, properties)


@app.on_event("startup")
async def startup():
    if db.is_closed():
        db.connect()

@app.on_event("shutdown")
async def shutdown():
    if not db.is_closed():
        db.close()




@app.get("/")
def read_root():
    return {
          "labels": ['100','200','300','400','500','600','700'], #datetime
          "data": [100,120, 140, 30, 10, 55, 40],
    }



@app.get("/get_info/{id}")
def get_info(id:int):

    sql =f"SELECT * FROM plant_features WHERE green_id = {id};"
    
    result = db.execute_sql(sql)
    data = {}

    for item in result:
        id = item[0]
        green_id = item[1]
        farm_id = item[2]
        plant_loc = item[3]
        rgb_path = item[4]
        mean_rgb = item[5]
        std_rgb = item[6]
        noir_path = item[7]
        ndvi_path = item[8]
        mean_ndvi = item[9]
        std_ndvi = item[10]
        leaf_area_index = item[11]
        created_at = item[12]

        try:
            if len(ndvi_path.split("\\")) == 1: std_rgb[0]

            pack_url = {
                "ndvi": "http://192.168.1.100:8000/image/NDVI/" + ndvi_path.split("\\")[-1], 
                "noir": "http://192.168.1.100:8000/image/NIR/" +noir_path.split("\\")[-1],
                "rgb": "http://192.168.1.100:8000/image/RGB/" +rgb_path.split("\\")[-1]
            }
        except:
            pack_url = {
                "ndvi": "http://192.168.1.100:8000/image/NDVI/" + ndvi_path.split("/")[-1], 
                "noir": "http://192.168.1.100:8000/image/NIR/" +noir_path.split("/")[-1],
                "rgb": "http://192.168.1.100:8000/image/RGB/" +rgb_path.split("/")[-1]
            }

        data[plant_loc] = pack_url
        
    return data    
        

@app.get("/image/{folder}/{imagename}")
async def image(folder: str, imagename: str):
    path = os.path.join(os.getcwd(), "images", folder, imagename)    
    import cv2
    img = cv2.imread(path) 

    if str(type(img)) == "<class 'numpy.ndarray'>":
        return FileResponse(path)
    else:
        path = os.path.join(os.getcwd(), "images", 'Capture.PNG')     
        return FileResponse(path)


if __name__ == '__main__':
    # from pyngrok import ngrok
    # ngrok.set_auth_token("2DLnTxYRlvbuVqwsPAU8bGquuF5_2nQPi5NNk88h8WmdPR9CG")
    # http_tunnel = ngrok.connect(8000,"http")
    # print(http_tunnel)

    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)


