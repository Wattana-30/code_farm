from sqlite3 import connect
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "*",
   
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def read_root():
    return {
          "labels": ['100','200','300','400','500','600','700'], #datetime
          "data": [100,120, 140, 30, 10, 55, 40],
    }


if __name__=='__main__':
    
    uvicorn.run("app:app",host='0.0.0.0',reload = True)
    
    


