import pandas as pd
import numpy as np
import os
import joblib
import json
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# --- CẤU HÌNH ---
DATA_DIR = 'data'
SAVE_DIR_ROOT = 'Save_model_ARIMA'

def train_arima_model(ticker):
    """
    Huấn luyện model ARIMA, đánh giá và dự báo tương lai.
    """
    try:
        print(f"🔄 Bắt đầu xử lý ARIMA cho: {ticker}")
        
        # 1. Định nghĩa đường dẫn
        data_path = os.path.join(DATA_DIR, f'{ticker}.csv')
        save_dir = os.path.join(SAVE_DIR_ROOT, ticker)
        os.makedirs(save_dir, exist_ok=True)

        if not os.path.exists(data_path):
            return f"❌ Không tìm thấy file dữ liệu: {data_path}"

        # 2. Đọc và xử lý dữ liệu
        df = pd.read_csv(data_path)
        
        # Xử lý cột Date linh hoạt
        col_date = 'Date' if 'Date' in df.columns else 'Ngay'
        df[col_date] = pd.to_datetime(df[col_date])
        df.set_index(col_date, inplace=True)
        
        # Chỉ lấy dữ liệu giá đóng cửa
        data = df[['Close']].copy().sort_index()
        
        # 3. Chia Train/Test (80/20) để đánh giá độ chính xác
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        # 4. Tìm Model tối ưu (Auto ARIMA) trên tập Train
        # seasonal=False vì dữ liệu ngày thường nhiễu, khó bắt chu kỳ năm
        model = auto_arima(train_data['Close'], 
                           start_p=1, start_q=1,
                           max_p=5, max_q=5,
                           d=None, seasonal=False, 
                           trace=False, error_action='ignore', 
                           suppress_warnings=True, stepwise=True)

        # 5. Đánh giá Model trên tập Test
        prediction = model.predict(n_periods=len(test_data))
        
        # Tính toán sai số
        metrics = {}
        if len(test_data) > 0:
            rmse = np.sqrt(mean_squared_error(test_data['Close'], prediction))
            mae = mean_absolute_error(test_data['Close'], prediction)
            mape = mean_absolute_percentage_error(test_data['Close'], prediction) * 100
            
            metrics = {
                "RMSE": round(rmse, 2),
                "MAE": round(mae, 2),
                "MAPE": round(mape, 2)
            }
        
        # 6. Retrain trên TOÀN BỘ dữ liệu để dự báo tương lai
        model_full = auto_arima(data['Close'], seasonal=False, stepwise=True, suppress_warnings=True)
        
        # Dự báo 30 ngày tới
        n_days = 30
        future_forecast, future_conf = model_full.predict(n_periods=n_days, return_conf_int=True)
        
        # Tạo ngày tương lai
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
        
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Close': future_forecast, # Đặt tên là Close để khớp với History
            'Lower': future_conf[:, 0],
            'Upper': future_conf[:, 1],
            'Type': 'Forecast'
        })

        # 7. Lưu dữ liệu
        # a. Lưu Model
        joblib.dump(model_full, os.path.join(save_dir, f'{ticker}_arima_model.pkl'))
        
        # b. Lưu Metrics
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
            
        # c. Lưu Dashboard Data (Gộp History + Forecast)
        history_df = data.copy()
        history_df['Date'] = history_df.index
        history_df = history_df.reset_index(drop=True)
        history_df['Type'] = 'History'
        history_df['Lower'] = np.nan
        history_df['Upper'] = np.nan
        
        # Chỉ lấy các cột cần thiết
        history_df = history_df[['Date', 'Close', 'Type']]
        future_save = future_df[['Date', 'Close', 'Type']]
        
        final_df = pd.concat([history_df, future_save], ignore_index=True)
        final_df.to_csv(os.path.join(save_dir, 'dashboard_data.csv'), index=False)

        return f"✅ Đã train xong ARIMA cho {ticker}. MAPE: {metrics.get('MAPE')}%"

    except Exception as e:
        return f"❌ Lỗi ARIMA mã {ticker}: {str(e)}"