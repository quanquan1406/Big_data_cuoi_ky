import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.backend import clear_session

# --- CẤU HÌNH ---
DATA_DIR = 'data'
SAVE_DIR_ROOT = 'Save_model_LSTM'
PAST_RANGE = 60  # Time step

def train_lstm_model(ticker):
    """
    Huấn luyện model LSTM.
    """
    clear_session() # Quan trọng: Xóa rác bộ nhớ cũ
    try:
        print(f"🔄 Đang bắt đầu train LSTM cho {ticker}...")
        
        # 1. Setup đường dẫn
        data_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        stock_model_dir = os.path.join(SAVE_DIR_ROOT, ticker)
        os.makedirs(stock_model_dir, exist_ok=True)

        if not os.path.exists(data_path):
            return f"❌ Không tìm thấy dữ liệu: {data_path}"

        # 2. Load Data
        stock = pd.read_csv(data_path)
        col_date = 'Date' if 'Date' in stock.columns else 'Ngay'
        stock[col_date] = pd.to_datetime(stock[col_date])
        stock.set_index(col_date, inplace=True)
        stock.sort_index(ascending=True, inplace=True)
        
        stock_lstm = stock[['Close']].copy()
        dataset = stock_lstm.values

        # 3. Scale Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Lưu Scaler ngay
        joblib.dump(scaler, os.path.join(stock_model_dir, "LSTM_scaler.pkl"))

        # 4. Prepare Train Data
        training_data_len = int(len(dataset) * 0.8)
        train = dataset[0:training_data_len, :]
        valid = dataset[training_data_len:, :]

        x_train, y_train = [], []
        for i in range(PAST_RANGE, len(train)):
            x_train.append(scaled_data[i-PAST_RANGE:i, 0])
            y_train.append(scaled_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # 5. Build & Train Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0) # verbose=0 để ẩn log

        # Lưu Model
        model.save(os.path.join(stock_model_dir, "LSTM.h5"))

        # 6. Evaluate & Save Results
        inputs = stock_lstm[len(stock_lstm) - len(valid) - PAST_RANGE:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = []
        for i in range(PAST_RANGE, inputs.shape[0]):
            X_test.append(inputs[i-PAST_RANGE:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        closing_price = model.predict(X_test)
        preds = scaler.inverse_transform(closing_price)

        y_valid = valid[:, 0]
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        mae = mean_absolute_error(y_valid, preds)
        r2 = r2_score(y_valid, preds)

        # Lưu Metrics json
        loss_detail = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
        metrics_path = os.path.join(stock_model_dir, "model_loss.json")
        
        full_metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                full_metrics = json.load(f)
        
        full_metrics["LSTM"] = loss_detail
        with open(metrics_path, 'w') as f:
            json.dump(full_metrics, f, indent=4)

        # Lưu CSV kết quả
        result_df = pd.DataFrame({'Actual': y_valid.flatten(), 'Prediction': preds.flatten()})
        result_df['Date'] = stock_lstm.index[training_data_len:]
        result_df.to_csv(os.path.join(stock_model_dir, "lstm_result.csv"), index=False)

        return f"✅ Đã train xong LSTM cho {ticker}. R2 Score: {r2:.4f}"

    except Exception as e:
        return f"❌ Lỗi LSTM mã {ticker}: {str(e)}"