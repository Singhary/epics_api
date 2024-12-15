import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
import pytz
from fastapi import FastAPI, HTTPException
from utils import (
    get_current_weather,
    predict_future,
    prepare_data,
    prepare_regression_data,
    read_historical_data,
    train_rain_model,
    train_regression_model,
)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/weather-prediction/{city}")
async def weather_prediction(city: str):
    try:
        current_weather = get_current_weather(city)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "weather_EPICS.csv")
        
        historical_data = read_historical_data(csv_path)

        x, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(x, y)
        wind_deg = current_weather["wind_gust_dir"] % 360
        compass_points = [
            ("N", 0, 11.25),
            ("NNE", 11.25, 33.75),
            ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75),
            ("E", 78.75, 101.25),
            ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25),
            ("SSE", 146.25, 168.75),
            ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75),
            ("SW", 213.75, 236.25),
            ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25),
            ("WNW", 281.25, 303.75),
            ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75),
            ("N", 348.75, 360),
        ]
        compass_direction = next(
            point for point, start, end in compass_points if start <= wind_deg < end
        )
        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1  # type: ignore

        current_data = {
            "MinTemp": current_weather["temp_min"],
            "MaxTemp": current_weather["temp_max"],
            "WindGustDir": compass_direction_encoded,
            "WindGustSpeed": current_weather["WindGustSpeed"],
            "Humidity": current_weather["humidity"],
            "Pressure": current_weather["pressure"],
            "Temp": current_weather["current_temp"],
        }
        current_df = pd.DataFrame([current_data])
        rain_prediction = rain_model.predict(current_df)[0]
        x_temp, y_temp = prepare_regression_data(historical_data, "Temp")
        x_humidity, y_humidity = prepare_regression_data(historical_data, "Humidity")
        temp_model = train_regression_model(x_temp, y_temp)
        humidity_model = train_regression_model(x_humidity, y_humidity)

        future_temp = predict_future(temp_model, current_weather["current_temp"])
        future_humidity = predict_future(humidity_model, current_weather["humidity"])
        timezone = pytz.timezone("Asia/Kolkata")
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

        future_times = [
            (next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(0, 5)
        ]

        time1, time2, time3, time4, time5 = future_times
        temp1, temp2, temp3, temp4, temp5 = future_temp
        hum1, hum2, hum3, hum4, hum5 = future_humidity

        return {
            "location": city,
            "current_temp": current_weather["current_temp"],
            "MinTemp": current_weather["temp_min"],
            "MaxTemp": current_weather["temp_max"],
            "feels_like": current_weather["feels_like"],
            "humidity": current_weather["humidity"],
            "clouds": current_weather["clouds"],
            "description": current_weather["description"],
            "city": current_weather["city"],
            "country": current_weather["country"],
            "time": datetime.now().strftime("%H:%M"),
            "date": datetime.now().strftime("%d-%m-%Y"),
            "wind": current_weather["WindGustSpeed"],
            "pressure": current_weather["pressure"],
            "visibility": current_weather["visibility"],
            "time1": time1,
            "time2": time2,
            "time3": time3,
            "time4": time4,
            "time5": time5,
            "temp1": f"{round(temp1, 1)}°C",
            "temp2": f"{round(temp2, 1)}°C",
            "temp3": f"{round(temp3, 1)}°C",
            "temp4": f"{round(temp4, 1)}°C",
            "temp5": f"{round(temp5, 1)}°C",
            "hum1": f"{round(hum1,1)}%",
            "hum2": f"{round(hum2,1)}%",
            "hum3": f"{round(hum3,1)}%",
            "hum4": f"{round(hum4,1)}%",
            "hum5": f"{round(hum5,1)}%",
            "rain_prediction": "Yes" if rain_prediction == 1 else "No",
        }

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
