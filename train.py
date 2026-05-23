import requests
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# BINANCE API
url = "https://api.binance.com/api/v3/klines"

params = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "limit": 1000
}



# GET DATA
response = requests.get(url, params=params)

data = response.json()



# DATAFRAME
df = pd.DataFrame(data)



# SELECT COLUMNS
df = df[[1,2,3,4,5]]

df.columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]



# CONVERT TO NUMBER
for col in df.columns:
    df[col] = df[col].astype(float)



# TARGET
df["target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)



# FEATURES
X = df[[
    "Close",
    "Volume"
]]



# LABEL
y = df["target"]



# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)



# MODEL
model = RandomForestClassifier()



# TRAIN
model.fit(X_train, y_train)



# PREDICT
predictions = model.predict(X_test)



# ACCURACY
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
