import pandas as pd
import numpy as np
import os
import joblib
import json
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Cấu hình đường dẫn
DATA_DIR = 'data'
SAVE_DIR_ROOT = 'Save_model_ARIMA'

def train_arima_model(ticker):
    try:
        # --- BƯỚC 1: CHUẨN HÓA TÊN MÃ (QUAN TRỌNG) ---
        # Nếu ticker là "BID" -> tự sửa thành "BID.VN"
        ticker = ticker.strip().upper()
        if not ticker.endswith('.VN'):
            ticker = f"{ticker}.VN"
            
        print(f"🔄 Bắt đầu xử lý ARIMA cho: {ticker}")
        
        # --- BƯỚC 2: KIỂM TRA DỮ LIỆU TRƯỚC ---
        data_path = os.path.join(DATA_DIR, f'{ticker}.csv')
        
        if not os.path.exists(data_path):
            return f"❌ Lỗi: Không tìm thấy file dữ liệu tại {data_path}. Hãy chạy tải lại dữ liệu."

        # --- BƯỚC 3: ĐỌC DỮ LIỆU ---
        df = pd.read_csv(data_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        data = df[['Close']].copy().sort_index()
        
        if len(data) < 30:
            return f"❌ Dữ liệu quá ít ({len(data)} dòng) để train model."

        # --- BƯỚC 4: TẠO THƯ MỤC LƯU (CHỈ TẠO KHI ĐÃ CÓ DATA) ---
        save_dir = os.path.join(SAVE_DIR_ROOT, ticker)
        os.makedirs(save_dir, exist_ok=True)

        # --- BƯỚC 5: TRAIN MODEL ---
        # Chia train/test
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        # Auto ARIMA trên tập train
        model_test = auto_arima(train_data['Close'], start_p=1, start_q=1, max_p=5, max_q=5,
                                seasonal=False, stepwise=True, suppress_warnings=True)
        
        prediction = model_test.predict(n_periods=len(test_data))
        
        metrics = {}
        if len(test_data) > 0:
            rmse = np.sqrt(mean_squared_error(test_data['Close'], prediction))
            mae = mean_absolute_error(test_data['Close'], prediction)
            mape = mean_absolute_percentage_error(test_data['Close'], prediction) * 100
            metrics = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE": round(mape, 2)}
        
        # Retrain Full để dự báo tương lai
        model_full = auto_arima(data['Close'], seasonal=False, stepwise=True, suppress_warnings=True)
        
        n_days = 30
        future_forecast, future_conf = model_full.predict(n_periods=n_days, return_conf_int=True)
        
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
        
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Close': future_forecast,
            'Lower': future_conf[:, 0],
            'Upper': future_conf[:, 1],
            'Type': 'Forecast'
        })

        # --- BƯỚC 6: LƯU FILE ---
        joblib.dump(model_full, os.path.join(save_dir, 'arima.pkl'))
        
        with open(os.path.join(save_dir, 'arima.json'), 'w') as f:
            json.dump(metrics, f)
            
        history_df = data.copy()
        history_df['Date'] = history_df.index
        history_df = history_df.reset_index(drop=True)
        history_df['Type'] = 'History'
        history_df['Lower'] = np.nan
        history_df['Upper'] = np.nan
        
        final_df = pd.concat([history_df[['Date', 'Close', 'Type']], 
                              future_df[['Date', 'Close', 'Type']]], ignore_index=True)
        
        final_df.to_csv(os.path.join(save_dir, 'arima.csv'), index=False)

        return f"✅ Đã train xong ARIMA cho {ticker}. Thư mục: {save_dir}"

    except Exception as e:
        return f"❌ Lỗi ARIMA mã {ticker}: {str(e)}"