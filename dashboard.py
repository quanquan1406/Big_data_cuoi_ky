import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Thêm những dòng này vào phần import trên cùng
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json
import numpy as np
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Xu hướng & Hiệu suất", 
    "📅 Phân tích Chu kỳ (Mới)", 
    "⚠️ Rủi ro & Biến động", 
    "🎯 Tương quan & Ranking",
    "🔮 Dự báo tương lai"
])

# --- TAB 1: XU HƯỚNG & HIỆU SUẤT ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Diễn biến giá hàng ngày")
        st.line_chart(df_filtered, height=400)
        
    with col2:
        st.subheader("🏆 Xếp hạng Tăng trưởng Tổng")
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
    
    st.subheader("📊 Lợi nhuận chi tiết từng năm (Grouped Bar Chart)")
    
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
    st.header("🔍 Hiệu ứng Mùa vụ (Seasonality)")
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

# --- TAB 5: DỰ BÁO AI (MỚI) ---
with tab5:
    st.header("🤖 Mô hình Dự báo Giá (LSTM)")
    
    # 1. Chọn ngân hàng cần dự báo (Chỉ lấy ngân hàng đầu tiên trong list đã chọn)
    target_bank = selected_banks[0]
    st.info(f"Đang chạy mô hình dự báo cho mã: **{target_bank}**")
    
    # Đường dẫn đến file mô hình (Cấu trúc: Save_model/ACB.VN/...)
    # Lưu ý: Thêm đuôi .VN nếu tên thư mục của bạn có .VN
    model_folder = f"Save_model/{target_bank}.VN" 
    
    model_path = os.path.join(model_folder, "LSTM.h5")
    scaler_path = os.path.join(model_folder, "LSTM_scaler.pkl")
    loss_path = os.path.join(model_folder, "model_loss.json")
    
    # Kiểm tra xem file có tồn tại không
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            # --- LOAD MÔ HÌNH ---
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            
            
            # --- HIỂN THỊ ĐÁNH GIÁ MÔ HÌNH (METRICS) ---
            col_ai1, col_ai2 = st.columns([1, 1]) # Chia đôi màn hình
            
            with col_ai1:
                st.subheader("📊 Hiệu quả mô hình (Evaluation Metrics)")
                
                if os.path.exists(loss_path):
                    with open(loss_path, 'r') as f:
                        metrics_data = json.load(f)
                    
                    # File json của bạn có dạng: {"LSTM": {"rmse": ..., "mae": ..., "r2": ...}}
                    if "LSTM" in metrics_data:
                        data = metrics_data["LSTM"]
                        
                        # Hiển thị 3 chỉ số quan trọng
                        m1, m2, m3 = st.columns(3)
                        
                        with m1:
                            st.metric(label="R2 Score (Độ phù hợp)", 
                                      value=f"{data.get('r2', 0):.4f}", 
                                      help="Càng gần 1 càng tốt")
                        
                        with m2:
                            st.metric(label="RMSE (Sai số)", 
                                      value=f"{data.get('rmse', 0):.0f}", 
                                      help="Càng thấp càng tốt")
                                      
                        with m3:
                            st.metric(label="MAE (Sai số tuyệt đối)", 
                                      value=f"{data.get('mae', 0):.0f}")
                        
                        # Đánh giá bằng lời văn
                        r2 = data.get('r2', 0)
                        if r2 > 0.9:
                            st.success("✅ Mô hình có độ chính xác RẤT CAO (>90%)")
                        elif r2 > 0.7:
                            st.info("ℹ️ Mô hình có độ chính xác KHÁ (>70%)")
                        else:
                            st.warning("⚠️ Mô hình có độ chính xác THẤP. Cần train lại.")
                            
                    else:
                        st.warning("File JSON không chứa key 'LSTM'.")
                        st.json(metrics_data) # In file ra để debug nếu cần
                else:
                    st.warning("Không tìm thấy file model_loss.json")

            # --- THỰC HIỆN DỰ BÁO ---
            with col_ai2:
                st.subheader("🔮 Dự báo ngày tiếp theo")
                
                # Lấy dữ liệu 60 ngày gần nhất của mã đó để dự báo
                # QUAN TRỌNG: time_step phải khớp với lúc bạn train mô hình (thường là 60)
                time_step = 60 
                
                # Lấy dữ liệu giá đóng cửa (hoặc giá điều chỉnh tùy lúc train bạn dùng cột nào)
                # Ở đây giả sử bạn train bằng cột Adj Close (Gia_Dieu_Chinh)
                data_last_60 = df[target_bank].values[-time_step:]
                
                # Reshape và Scale dữ liệu
                data_last_60 = data_last_60.reshape(-1, 1)
                data_scaled = scaler.transform(data_last_60)
                
                # Reshape cho đúng input của LSTM (1, 60, 1)
                X_input = data_scaled.reshape(1, time_step, 1)
                
                # Dự báo
                pred_scaled = model.predict(X_input)
                pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                
                # Lấy giá ngày gần nhất để so sánh
                last_price = df[target_bank].iloc[-1]
                change = pred_price - last_price
                pct_change = (change / last_price) * 100
                
                # Hiển thị kết quả kiểu số lớn (Metric)
                st.metric(
                    label=f"Giá dự báo ngày mai ({target_bank})",
                    value=f"{pred_price:,.0f} VND",
                    delta=f"{change:,.0f} VND ({pct_change:.2f}%)"
                )
                
                st.write(f"Giá đóng cửa gần nhất: **{last_price:,.0f} VND**")
                
                if pct_change > 0:
                    st.success("Mô hình dự báo: **TĂNG** 🚀")
                else:
                    st.error("Mô hình dự báo: **GIẢM** 📉")

        except Exception as e:
            st.error(f"Lỗi khi chạy mô hình: {e}")
            st.warning("Gợi ý: Kiểm tra xem 'time_step' (số ngày lùi lại) trong code dashboard có khớp với lúc bạn train mô hình không?")
            
    else:
        st.warning(f"⚠️ Chưa tìm thấy mô hình đã lưu cho mã **{target_bank}**.")
        st.write(f"Vui lòng kiểm tra thư mục: `{model_folder}`")
        st.write("Cấu trúc file cần thiết: `LSTM.h5`, `LSTM_scaler.pkl`")