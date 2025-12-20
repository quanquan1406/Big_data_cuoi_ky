import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json
import numpy as np
import data as dt_process
import train_arima
import train_lstm
import time
# -----------------------------------------------------------
# 1. CẤU HÌNH & HÀM TẢI DỮ LIỆU
# -----------------------------------------------------------
st.set_page_config(layout="wide", page_title="Bank Stock Analysis Dashboard")

# Custom CSS để làm đẹp tiêu đề
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Dashboard Phân Tích Cổ Phiếu Ngân Hàng")
st.markdown("---")

@st.cache_data
def load_data():
    folder_path = 'data' 
    df_merged = pd.DataFrame()
    
    if not os.path.exists(folder_path):
        return None
        
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            df_temp = pd.read_csv(file_path)
            col_date = 'Date' if 'Date' in df_temp.columns else 'Ngay'
            df_temp[col_date] = pd.to_datetime(df_temp[col_date])
            df_temp.set_index(col_date, inplace=True)
            
            ticker = file.split('.')[0].replace('.VN','')
            
            if 'Adj Close' in df_temp.columns:
                df_merged[ticker] = df_temp['Adj Close']
            elif 'Gia_Dieu_Chinh' in df_temp.columns:
                df_merged[ticker] = df_temp['Gia_Dieu_Chinh']
            else:
                df_merged[ticker] = df_temp['Close']
        except:
            continue
            
    df_merged.dropna(inplace=True)
    return df_merged

df = load_data()

if df is None:
    st.error("Lỗi: Không tìm thấy thư mục 'data'.")
    st.stop()

# -----------------------------------------------------------
# 2. SIDEBAR (BỘ LỌC)
# -----------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình dữ liệu")
    all_banks = df.columns.tolist()
    selected_banks = st.multiselect("Chọn ngân hàng:", all_banks, default=all_banks[:5])
    
    start_date = df.index.min()
    end_date = df.index.max()
    date_range = st.date_input("Khoảng thời gian:", [start_date, end_date])

if not selected_banks:
    st.warning("Vui lòng chọn ít nhất 1 ngân hàng.")
    st.stop()

# Lọc dữ liệu
df_filtered = df[selected_banks]
df_filtered = df_filtered[(df_filtered.index >= pd.to_datetime(date_range[0])) & 
                          (df_filtered.index <= pd.to_datetime(date_range[1]))]
df_returns = df_filtered.pct_change().dropna()

# -----------------------------------------------------------
# 3. GIAO DIỆN CHÍNH (CÁC TAB)
# -----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Xu hướng & Hiệu suất", 
    "📅 Phân tích Chu kỳ (Mới)", 
    "⚠️ Rủi ro & Biến động", 
    "🎯 Tương quan & Ranking",
    "🔮 Dự báo tương lai",
    "⚙️ Control Panel"
])

# --- TAB 1: XU HƯỚNG & HIỆU SUẤT ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Diễn biến giá hàng ngày")
        st.line_chart(df_filtered, height=400)
        
    with col2:
        st.subheader("Xếp hạng Tăng trưởng Tổng")
        # Tính tổng tăng trưởng
        total_return = (df_filtered.iloc[-1] / df_filtered.iloc[0] - 1) * 100
        total_return = total_return.sort_values(ascending=True)
        
        # Vẽ bằng Matplotlib để giữ màu Xanh/Đỏ
        fig_rank, ax_rank = plt.subplots(figsize=(4, 6))
        colors = ['red' if x < 0 else 'green' for x in total_return]
        total_return.plot(kind='barh', color=colors, alpha=0.7, ax=ax_rank)
        ax_rank.set_xlabel("% Tăng trưởng")
        ax_rank.grid(axis='x', linestyle='--', alpha=0.5)
        st.pyplot(fig_rank)

    st.divider()
    
    st.subheader("Lợi nhuận chi tiết từng năm (Grouped Bar Chart)")
    
    # 1. Chuẩn bị dữ liệu
    yearly_ret = df_filtered.resample('YE').apply(lambda x: x.iloc[-1] / x.iloc[0] - 1) * 100
    yearly_ret.index = yearly_ret.index.year
    
    # Chuyển dữ liệu từ dạng Rộng (Wide) sang dạng Dài (Long) để vẽ Plotly
    yearly_long = yearly_ret.reset_index().melt(id_vars='Date', var_name='Bank', value_name='Return')
    yearly_long.rename(columns={'Date': 'Năm', 'Bank': 'Ngân hàng', 'Return': 'Lợi nhuận (%)'}, inplace=True)
    
    # 2. Vẽ bằng Plotly Express
    import plotly.express as px
    
    fig_grouped = px.bar(
        yearly_long, 
        x="Năm", 
        y="Lợi nhuận (%)", 
        color="Ngân hàng", 
        barmode="group",  # <--- QUAN TRỌNG: Lệnh này giúp các cột đứng cạnh nhau
        text_auto='.1f',  # Hiển thị số trên đầu cột (1 chữ số thập phân)
        color_discrete_sequence=px.colors.qualitative.Prism # Chọn bảng màu đẹp, rõ ràng
    )
    
    # Tinh chỉnh giao diện biểu đồ
    fig_grouped.update_layout(
        xaxis=dict(tickmode='linear'), # Đảm bảo hiện đủ các năm 2020, 2021...
        legend_title_text='Mã CP',
        height=500
    )
    
    # Hiển thị lên Streamlit
    st.plotly_chart(fig_grouped, use_container_width=True)

# --- TAB 2: PHÂN TÍCH CHU KỲ (MỚI) ---
with tab2:
    st.header("Hiệu ứng Mùa vụ (Seasonality)")
    st.markdown("Biểu đồ này cho biết lợi nhuận trung bình của các ngân hàng theo từng tháng trong năm.")
    
    # Tính trung bình theo tháng
    df_temp_ret = df_returns.copy()
    df_temp_ret['Month'] = df_temp_ret.index.month
    monthly_seasonality = df_temp_ret.groupby('Month').mean() * 100 # Ra %
    
    # Vẽ biểu đồ
    st.bar_chart(monthly_seasonality)
    
    st.info("💡 **Gợi ý:** Nếu cột tháng 1, 2 cao -> Có hiệu ứng tăng giá dịp Tết. Nếu tháng 5 thấp -> Hiệu ứng 'Sell in May'.")

# --- TAB 3: RỦI RO ---
with tab3:
    col_risk1, col_risk2 = st.columns(2)
    
    with col_risk1:
        st.subheader("Phân phối lợi nhuận (Boxplot)")
        fig_box, ax_box = plt.subplots()
        sns.boxplot(data=df_returns * 100, ax=ax_box, palette="Set3")
        ax_box.set_ylabel("Lợi nhuận ngày (%)")
        st.pyplot(fig_box)
        
    with col_risk2:
        st.subheader("Mức sụt giảm kỷ lục (Max Drawdown)")
        rolling_max = df_filtered.cummax()
        drawdown = df_filtered / rolling_max - 1.0
        st.area_chart(drawdown)

# --- TAB 4: TƯƠNG QUAN ---
with tab4:
    col_corr1, col_corr2 = st.columns([1, 1])
    
    with col_corr1:
        st.subheader("Ma trận tương quan")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 8))
        sns.heatmap(df_returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)
        
    with col_corr2:
        st.subheader("Rủi ro vs Lợi nhuận")
        rets = df_returns.mean() * 252
        risk = df_returns.std() * (252 ** 0.5)
        
        fig_scat, ax_scat = plt.subplots(figsize=(8, 8))
        ax_scat.scatter(risk, rets, s=100, c='teal', alpha=0.6)
        for label, x, y in zip(rets.index, risk, rets):
            ax_scat.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', ha='center', weight='bold')
        ax_scat.set_xlabel("Rủi ro (Volatility)")
        ax_scat.set_ylabel("Lợi nhuận (Return)")
        ax_scat.grid(True, linestyle='--')
        st.pyplot(fig_scat)

# --- TAB 5: DỰ BÁO TƯƠNG LAI ---
with tab5:
    st.header("🔮 So sánh Mô hình Dự báo")
    
    target_bank = selected_banks[0] # Lấy ngân hàng đầu tiên trong danh sách chọn
    st.caption(f"Đang hiển thị dữ liệu dự báo cho: **{target_bank}**")

    # Tạo 2 tab con
    sub_tab_arima, sub_tab_lstm = st.tabs(["📈 ARIMA (Thống kê)", "🧠 LSTM (Deep Learning)"])

    # =========================================================
    # 1. SUB-TAB ARIMA
    # =========================================================
    with sub_tab_arima:
        arima_folder = f"Save_model_ARIMA/{target_bank}.VN"
        csv_path = os.path.join(arima_folder, "arima.csv")
        metrics_path = os.path.join(arima_folder, "arima.json")
        
        if os.path.exists(csv_path) and os.path.exists(metrics_path):
            # Load Metrics
            with open(metrics_path, 'r') as f:
                arima_metrics = json.load(f)
            
            st.subheader("1. Độ chính xác (Test Set)")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{arima_metrics.get('RMSE', 0):.2f}")
            c2.metric("MAE", f"{arima_metrics.get('MAE', 0):.2f}")
            c3.metric("MAPE", f"{arima_metrics.get('MAPE', 0):.2f}%")
            
            # Load Data & Plot
            df_arima = pd.read_csv(csv_path)
            st.subheader("2. Biểu đồ Dự báo")
            fig_arima = px.line(df_arima, x='Date', y='Close', color='Type',
                                title=f"Dự báo ARIMA - {target_bank}",
                                color_discrete_map={"History": "#A6CEE3", "Forecast": "#E31A1C"})
            st.plotly_chart(fig_arima, use_container_width=True)
        else:
            st.warning(f"⚠️ Chưa có dữ liệu ARIMA cho {target_bank}. Hãy kiểm tra thư mục: {arima_folder}")

    # =========================================================
    # 2. SUB-TAB LSTM 
    # =========================================================
    with sub_tab_lstm:
        # SỬA LỖI TYPO: LSMT -> LSTM
        lstm_folder = f"Save_model_LSTM/{target_bank}.VN" 
        
        model_path = os.path.join(lstm_folder, "lstm.h5")
        scaler_path = os.path.join(lstm_folder, "lstm.pkl")
        loss_path = os.path.join(lstm_folder, "lstm.json")
        result_csv_path = os.path.join(lstm_folder, "lstm.csv")

        # Kiểm tra file trước khi load
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                # Load model (Nên dùng caching nếu có thể, ở đây load trực tiếp)
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                
                # 1. Hiển thị Metrics
                col_m, col_p = st.columns(2)
                with col_m:
                    st.subheader("1. Đánh giá Model")
                    if os.path.exists(loss_path):
                        with open(loss_path, 'r') as f:
                            metrics = json.load(f).get("LSTM", {})
                        st.write(f"**R2 Score:** {metrics.get('r2', 'N/A')}")
                        st.write(f"**RMSE:** {metrics.get('rmse', 'N/A')}")
                    else:
                        st.info("Không tìm thấy file metrics.")

                # 2. Dự báo ngày tiếp theo
                with col_p:
                    st.subheader("2. Dự báo ngày mai")
                    # Lấy 60 ngày cuối từ dữ liệu gốc
                    if target_bank in df.columns:
                        last_60_days = df[target_bank].values[-60:].reshape(-1, 1)
                        last_60_scaled = scaler.transform(last_60_days)
                        X_test = last_60_scaled.reshape(1, 60, 1)
                        
                        pred_scaled = model.predict(X_test)
                        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                        current_price = df[target_bank].iloc[-1]
                        
                        delta = pred_price - current_price
                        st.metric("Giá dự kiến", f"{pred_price:,.0f} VND", f"{delta:,.0f} VND")

                # 3. Vẽ biểu đồ so sánh (Actual vs Predict)
                st.subheader("3. Kết quả chạy thực nghiệm")
                if os.path.exists(result_csv_path):
                    df_res = pd.read_csv(result_csv_path)
                    # Melt data cho Plotly
                    df_melt = df_res.melt(id_vars='Date', value_vars=['Actual', 'Prediction'], 
                                          var_name='Legend', value_name='Price')
                    
                    fig_lstm = px.line(df_melt, x='Date', y='Price', color='Legend',
                                       color_discrete_map={"Actual": "gray", "Prediction": "#00CC96"})
                    st.plotly_chart(fig_lstm, use_container_width=True)
                else:
                    st.warning("Chưa có file kết quả test (lstm_result.csv).")

            except Exception as e:
                st.error(f"Lỗi khi chạy model LSTM: {e}")
        else:
            st.info(f"⚠️ Chưa train model LSTM cho mã {target_bank}")


# --- TAB 6: CONTROL PANEL ---
with tab6:
    st.header("⚙️ Hệ thống Quản trị Dữ liệu & Mô hình")
    
    col1, col2 = st.columns(2)
    
    # --- CỘT 1: DỮ LIỆU ---
    with col1:
        st.subheader("1. Cập nhật Dữ liệu")
        st.info("Nhấn nút bên dưới để tải dữ liệu mới nhất từ thị trường và làm sạch.")
        
        # Nút Tải Data
        if st.button("📥 Tải Dữ liệu Mới (Raw)", use_container_width=True):
            with st.spinner('Đang kết nối API để tải dữ liệu... (Vui lòng chờ)'):
                try:
                    msg = dt_process.download_data() # Gọi hàm
                    st.success(msg)
                    time.sleep(1)
                    st.rerun() # Load lại trang để nhận data mới
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        # Nút Làm sạch Data
        if st.button("🧹 Làm sạch Dữ liệu (Clean)", use_container_width=True):
            with st.spinner('Đang xử lý Missing values & Outliers...'):
                try:
                    msg = dt_process.clean_data()
                    st.success(msg)
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    # --- CỘT 2: HUẤN LUYỆN MODEL ---
    with col2:
        st.subheader("2. Huấn luyện Mô hình (Retrain)")
        st.warning("⚠️ Lưu ý: Việc train model tốn nhiều tài nguyên và thời gian (đặc biệt là LSTM).")
        
        # Chọn mã để train lại (tránh train hết tất cả sẽ rất lâu)
        bank_to_train = st.selectbox("Chọn mã cổ phiếu cần Train lại:", all_banks)
        
        # Nút Train ARIMA
        if st.button(f"📈 Train ARIMA cho {bank_to_train}", use_container_width=True):
            with st.status(f"Đang train ARIMA cho {bank_to_train}...", expanded=True) as status:
                st.write("Đang đọc dữ liệu và tìm tham số tối ưu...")
                try:
                    # GỌI HÀM TỪ FILE train_arima.py
                    res = train_arima.train_arima_model(bank_to_train) 
                    
                    status.update(label="Hoàn tất!", state="complete", expanded=False)
                    st.success(res)
                except Exception as e:
                    status.update(label="Lỗi!", state="error")
                    st.error(f"Chi tiết: {e}")

        # Nút Train LSTM
        if st.button(f"🧠 Train LSTM cho {bank_to_train}", use_container_width=True):
            with st.spinner('Đang khởi tạo TensorFlow và huấn luyện (Mất khoảng 1-2 phút)...'):
                try:
                    # GỌI HÀM TỪ FILE train_lstm.py
                    res = train_lstm.train_lstm_model(bank_to_train)
                    
                    st.success(res)
                except Exception as e:
                    st.error(f"Lỗi: {e}")