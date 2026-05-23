import pickle
import numpy as np



# LOAD MODEL
model = pickle.load(
    open("btc_model.pkl", "rb")
)



# NEW DATA
# Close, Volume
new_data = np.array([
    [65000, 30000000000]
])



# PREDICT
prediction = model.predict(new_data)



# OUTPUT
if prediction[0] == 1:
    print("BTC may go UP")
else:
    print("BTC may go DOWN")
