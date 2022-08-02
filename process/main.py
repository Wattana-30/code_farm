import random
from paho.mqtt import client as mqtt_client
import segmentation as smt


broker = 'broker.emqx.io'
port = 1883
sub_topic = ["plant/plc-to-jetson"]
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 100)}'
username = 'emqx'
password = 'public'


def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        if msg.topic == "plant/plc-to-jetson":
            payload = msg.payload.decode()
            payloadList = payload.split(" ")
            if payloadList[2] == "ready":
                data_str = smt.startEvent(payloadList[0], payloadList[1])
                client.publish("plant/jetson-to-server",data_str)
                client.publish("plant/jetson-to-plc", "ready")

    for topic in sub_topic:
        client.subscribe(topic)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()    # Read image, create blank masks, color threshold


if __name__ == '__main__':
    run()
