# 🌦️ **Weather Prediction API**  

Welcome to the **Weather Prediction API**! This service, built using **FastAPI**, fetches real-time weather data and predicts future metrics like **temperature**, **humidity**, and the likelihood of **rain** for the next 5 hours. 🌤️  

🚀 **Deployed at**: [🌍 epics-api.onrender.com](https://epics-api.onrender.com/)

---

## 🚀 **Features**  

✨ **Real-time Weather**: Fetch live weather data for any city using the **OpenWeather API**.  
🔮 **5-Hour Predictions**: Predict **temperature** and **humidity** for the next 5 hours.  
🌧️ **Rain Prediction**: Determine the chance of rain using a trained ML model based on historical weather data.  
📊 **Structured Responses**: Clear and easy-to-use JSON responses.  

---

## 🌐 **API Endpoints**  

### 1. **GET /weather-prediction/{city}**  

**Description**: Fetch real-time weather data and predictions for a specific city.  

- **Method**: `GET`  
- **URL**: `https://epics-api.onrender.com/weather-prediction/{city}`  
- **Path Parameter**:  
  - `city` *(string)*: The name of the city to fetch weather predictions for.  

**Request Example**:  
```http
GET https://epics-api.onrender.com/weather-prediction/Mumbai
