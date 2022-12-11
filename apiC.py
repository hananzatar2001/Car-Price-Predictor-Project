from pydantic import BaseModel
from fastapi import FastAPI
from fastapi import Request
from http import HTTPStatus
from typing import Dict
import uvicorn
import pickle

modelknn=pickle.load(open('modelKN.pkl','rb'))

app = FastAPI()

@app.get('/')
def geet():
    """ health check """
    return {'message': 'Hi Hanan'}

                                                         
                                                          
                                                
class features(BaseModel):
    model: float
    motor_power: float
    passengers: float
    car_speedometer: float
    namecar:str
    color:str
    origin_car:str
    fuel_type:str
    glass:str
    car_license:str
    payment_method:str

@app.post("/predict")
def train_plynomial(req : features):
    model=req.model
    motor_power=req.motor_power
    car_speedometer=req.car_speedometer
    passengers=req.passengers
    namecar=req.namecar
    color=req.color
    origin_car=req.origin_car
    fuel_type=req.fuel_type
    glass=req.glass
    car_license=req.car_license
    payment_method=req.payment_method


    features =list([model,
               motor_power,
               car_speedometer,
               glass,
               passengers,
               namecar,
               color,
               origin_car,
               fuel_type,
               car_license,
               payment_method
                    ])
     
    prediction = modelknn.predict([[model,
                                 motor_power,
                                 car_speedometer,
                                 glass,
                                 passengers,
                                 namecar,
                                 color,
                                 origin_car,
                                 fuel_type,
                                 car_license,
                                 payment_method
                                 ]])
    return {'you can sell your house for {} '.format(prediction)}
    
if __name__=="__main__":
    uvicorn.run(app)