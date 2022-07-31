from contextvars import ContextVar
import peewee

USERNAME = "root"
PASSWORD = "232544da"
DATABASE_NAME = "plant_monitoring_robot"
HOST = "localhost"
PORT = 3306

db_state_default = {"closed": None, "conn": None, "ctx": None, "transactions": None}
db_state = ContextVar("db_state", default=db_state_default.copy())


class PeeweeConnectionState(peewee._ConnectionState):
    def __init__(self, **kwargs):
        super().__setattr__("_state", db_state)
        super().__init__(**kwargs)

    def __setattr__(self, name, value):
        self._state.get()[name] = value

    def __getattr__(self, name):
        return self._state.get()[name]


db = peewee.MySQLDatabase(
    DATABASE_NAME, 
    user=USERNAME, 
    password=PASSWORD, 
    host=HOST, 
    port=PORT, 
    # check_same_thread=False
)

db._state = PeeweeConnectionState()
