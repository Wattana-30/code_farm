from fastapi_mqtt import FastMQTT, MQTTConfig
from fastapi import FastAPI
import uvicorn
import time

# routers
from routers import users, items, plant_features, plant_qrcode, farm, greenhouse, constant

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
    mqtt.client.subscribe("plant/jetson-to-server")
    mqtt.client.subscribe("plant/plc-to-jetson")


@mqtt.on_message()
async def message(client, topic, payload, qos, properties):
    # print("Received message: ", topic, payload.decode(), qos, properties)
    # print("Received message: ", payload.decode())
    print("Topic: ", topic)

    if topic == 'plant/farm':
        greenEnv.pack_and_insert_greenhouse(payload.decode())
    elif topic == 'plant/plc-to-jetson':
        payload = payload.decode()
        payloadList = payload.split(" ")

        if payloadList[1] == "false": return

        if payloadList[2] == "ready":
            utils.startEvent(mqtt, payloadList[0], payloadList[1])
            client.publish("plant/jetson-to-plc", "ready")
            

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




from fastapi.responses import FileResponse

some_file_path = "images/NIR/Point_6_2022-08-12 00_27_45.jpg"

@app.get("/")
async def image():
    return FileResponse(some_file_path)


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)


