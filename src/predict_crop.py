
import joblib

model = joblib.load("xgboost_crop_model.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):

    input_data = [[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]]

    prediction = model.predict(input_data)

    return label_encoder.inverse_transform(
        prediction.astype(int)
    )[0]
