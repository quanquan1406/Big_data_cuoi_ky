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
    "📅 Phân tích Chu kỳ ", 
    "⚠️ Rủi ro & Biến động", 
    "🎯 Tương quan & Ranking",
    "🔮 Dự báo tương lai",
    "⚙️ Control Panel"
])

# --- TAB 1: XU HƯỚNG & HIỆU SUẤT ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Diễn biến giá hàng ngày")
        st.line_chart(df_filtered, height=400)
        
    with col2:
        st.subheader("2. Xếp hạng Tăng trưởng Tổng")
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
    
    st.subheader("3. Lợi nhuận chi tiết từng năm ")
    
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

# --- TAB 2: PHÂN TÍCH CHU KỲ  ---
with tab2:
    st.header("Hiệu ứng Mùa vụ (Monthly Seasonality)")
    st.markdown("Biểu đồ Heatmap dưới đây giúp bạn nhanh chóng nhận diện tháng nào trong năm thường mang lại lợi nhuận cao (Xanh) hoặc rủi ro giảm giá (Đỏ).")

    # 1. Xử lý dữ liệu
    df_seasonal = df_returns.copy()
    df_seasonal['Month'] = df_seasonal.index.month
    
    monthly_avg = df_seasonal.groupby('Month').mean() * 100
    
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    monthly_avg.index = monthly_avg.index.map(month_names)
    
    heatmap_data = monthly_avg.T 

    # 2. Chia cột
    col_chart, col_note = st.columns([3, 1])

    with col_chart:
        fig_heatmap = px.imshow(
            heatmap_data,
            labels=dict(x="Tháng", y="Mã CP", color="Lợi nhuận TB (%)"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale="RdBu",
            origin='lower'
        )
        
        # --- CẤU HÌNH TƯƠNG TÁC ---
        fig_heatmap.update_layout(
            title_text=None, 
            xaxis_title=None, 
            yaxis_title=None,
            height=500, 
            margin=dict(l=0, r=0, t=20, b=0),
            dragmode='pan' # Kéo chuột là di chuyển
        )
        
        # scrollZoom=True: Lăn chuột là Zoom
        st.plotly_chart(fig_heatmap, use_container_width=True, 
                        config={'displayModeBar': False, 'scrollZoom': True})

    with col_note:
        st.info("ℹ️ **Cách đọc biểu đồ:**")
        st.markdown("""
        * **Màu Xanh Đậm:** Tháng tăng giá mạnh.
        * **Màu Đỏ Đậm:** Tháng giảm điểm sâu.
        * **Màu Nhạt/Trắng:** Biến động thấp.
        
        **Quy luật tham khảo:**
        * *Hiệu ứng tháng Giêng (Jan Effect)*
        * *Sell in May (Tháng 5 thường đỏ)*
        * *Sóng báo cáo tài chính (Tháng 4, 7, 10)*
        """)


# --- TAB 3: RỦI RO & BIẾN ĐỘNG  ---
with tab3:
    st.header("Phân tích Rủi ro & Biến động giá")
    col_risk1, col_risk2 = st.columns(2)
    
    with col_risk1:
        st.subheader("1. Phân phối biến động (Boxplot)")
        fig_box, ax_box = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=df_returns * 100, ax=ax_box, palette="Set3")
        ax_box.set_ylabel("Biến động ngày (%)")
        st.pyplot(fig_box)
        
        # --- CHÚ THÍCH BOXPLOT ---
        with st.expander("ℹ️ Cách đọc biểu đồ Boxplot", expanded=True):
            st.caption("""
            * **Chiều dài hộp:** Hộp càng dài ➔ Giá cổ phiếu biến động càng mạnh (Rủi ro cao).
            * **Vạch giữa:** Mức lợi nhuận trung vị (thường gần 0).
            * **Các chấm tròn:** Là các ngày có biến động bất thường (tăng trần hoặc giảm sàn).
            """)
        
    with col_risk2:
        st.subheader("2. Mức sụt giảm kỷ lục (Max Drawdown)")
        rolling_max = df_filtered.cummax()
        drawdown = df_filtered / rolling_max - 1.0
        st.area_chart(drawdown, height=350)

        # --- CHÚ THÍCH DRAWDOWN ---
        with st.expander("ℹ️ Cách đọc Max Drawdown", expanded=True):
            st.caption("""
            * Biểu đồ này trả lời câu hỏi: **"Nếu tôi lỡ mua đúng đỉnh, thì tài khoản đang tạm lỗ bao nhiêu %?"**
            * **Càng đi xuống sâu:** Rủi ro càng lớn.
            * Cổ phiếu tốt là cổ phiếu có mức sụt giảm thấp và nhanh chóng hồi phục về mức 0 (vượt đỉnh cũ).
            """)

# --- TAB 4: TƯƠNG QUAN & RANKING  ---
with tab4:
    st.header("Tương quan & Vị thế Cổ phiếu")
    col_corr1, col_corr2 = st.columns([1, 1])
    
    with col_corr1:
        st.subheader("1. Ma trận tương quan (Correlation)")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 8))
        sns.heatmap(df_returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)
        
        # --- CHÚ THÍCH HEATMAP ---
        with st.expander("ℹ️ Cách đọc Ma trận Tương quan", expanded=True):
            st.caption("""
            * **Màu Đỏ (gần +1):** Hai mã cùng chiều (Mã A tăng, Mã B cũng tăng). 
              > *Rủi ro: Không nên mua cả 2 mã này cùng lúc vì danh mục sẽ không được đa dạng hóa.*
            * **Màu Xanh (gần -1 hoặc 0):** Ít liên quan hoặc ngược chiều.
              > *Tốt: Giúp giảm thiểu rủi ro cho danh mục đầu tư.*
            """)
        
    with col_corr2:
        st.subheader("2. Rủi ro vs Lợi nhuận (Risk-Return)")
        
        # Tính toán
        rets = df_returns.mean() * 252
        risk = df_returns.std() * (252 ** 0.5)
        
        mean_risk = risk.mean()
        mean_ret = rets.mean()
        
        fig_scat, ax_scat = plt.subplots(figsize=(8, 8))
        ax_scat.scatter(risk, rets, s=100, c='teal', alpha=0.7, edgecolors='black')
        
        # Đường trung bình
        ax_scat.axvline(x=mean_risk, color='red', linestyle='--', linewidth=1, label='TB Rủi ro')
        ax_scat.axhline(y=mean_ret, color='red', linestyle='--', linewidth=1, label='TB Lợi nhuận')
        
        for label, x, y in zip(rets.index, risk, rets):
            ax_scat.annotate(label, xy=(x, y), xytext=(5, 5), textcoords='offset points', ha='left', weight='bold')
            
        ax_scat.set_xlabel("Rủi ro (Volatility)")
        ax_scat.set_ylabel("Lợi nhuận kỳ vọng")
        ax_scat.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig_scat)

        # --- CHÚ THÍCH SCATTER ---
        with st.expander("ℹ️ Cách chọn Cổ phiếu 'Vàng'", expanded=True):
            st.caption("""
            Biểu đồ được chia làm 4 góc bởi 2 đường kẻ đỏ (Mức trung bình ngành):
            *  **Góc Trái-Trên (Lợi nhuận Cao - Rủi ro Thấp):** Cổ phiếu TỐT NHẤT.
            *  **Góc Phải-Trên (Lợi nhuận Cao - Rủi ro Cao):** Lợi nhuận cao đi kèm rủi ro lớn (High Risk High Return).
            *  **Góc Trái-Dưới (Lợi nhuận Thấp - Rủi ro Thấp):** Cổ phiếu AN TOÀN (Biến động ít, phù hợp nắm giữ dài hạn/phòng thủ).
            *  **Góc Phải-Dưới (Lợi nhuận Thấp - Rủi ro Cao):** Cổ phiếu KÉM HẤP DẪN (Nên cân nhắc kỹ trước khi đầu tư).
            """)

# --- TAB 5: DỰ BÁO TƯƠNG LAI  ---
with tab5:
    st.header("Chi tiết Dự báo & So sánh Mô hình")
    
    # Chia 3 cột
    col_sel, col_space, col_price = st.columns([1.5, 3.5, 1.5])
    
    with col_sel:
        target_bank = st.selectbox("📌 Chọn mã xem dự báo:", all_banks)
    
    with col_space:
        st.write("") 
    
    with col_price:
        if target_bank in df.columns:
            current_price = df[target_bank].iloc[-1]
            prev_price = df[target_bank].iloc[-2]
            delta_price = current_price - prev_price
            delta_pct = (delta_price / prev_price) * 100
            
            st.metric(
                label=f"Giá hiện tại ({target_bank})",
                value=f"{current_price:,.0f} VND",
                delta=f"{delta_price:,.0f} VND ({delta_pct:.2f}%)"
            )
        else:
            st.error("Không tìm thấy dữ liệu.")

    st.divider()

    sub_tab_arima, sub_tab_lstm = st.tabs(["📈 ARIMA (Thống kê)", "🧠 LSTM (Deep Learning)"])

    # 1. SUB-TAB ARIMA 
    with sub_tab_arima:
        arima_folder = f"Save_model_ARIMA/{target_bank}.VN"
        csv_path = os.path.join(arima_folder, "arima.csv")
        metrics_path = os.path.join(arima_folder, "arima.json")
        
        if os.path.exists(csv_path) and os.path.exists(metrics_path):
            # Load dữ liệu trước để tính toán cho Metric
            df_arima = pd.read_csv(csv_path)
            with open(metrics_path, 'r') as f:
                arima_metrics = json.load(f)
            
            st.markdown(f"##### 📊 Kết quả đánh giá & Dự báo - {target_bank}")
            
            # ---  ---
            m1, m2, m3, m4 = st.columns(4)
            
            # 3 Cột đầu: Chỉ số đánh giá
            m1.metric(
                "RMSE", 
                f"{arima_metrics.get('RMSE', 0):.2f}",
                help="Root Mean Squared Error: Sai số trung bình phương gốc.\n\nCàng THẤP càng tốt. Chỉ số này phạt nặng các sai số lớn (outliers)."
            )
            m2.metric(
                "MAE", 
                f"{arima_metrics.get('MAE', 0):.2f}",
                help="Mean Absolute Error: Sai số tuyệt đối trung bình.\n\nCàng THẤP càng tốt. Cho biết trung bình mô hình lệch bao nhiêu VND so với thực tế."
            )
            m3.metric(
                "MAPE", 
                f"{arima_metrics.get('MAPE', 0):.2f}%",
                help="Mean Absolute Percentage Error: Sai số phần trăm trung bình.\n\nCàng THẤP càng tốt. Ví dụ: 1.5% nghĩa là dự báo lệch khoảng 1.5% so với giá thật."
            )
            
            # Cột 4: DỰ BÁO NGÀY MAI (Tính toán từ file CSV)
            try:
                # Lấy giá thực tế gần nhất (dòng cuối cùng của History)
                last_history = df_arima[df_arima['Type'] == 'History'].iloc[-1]
                current_price_arima = last_history['Close']
                
                # Lấy giá dự báo ngày mai (dòng đầu tiên của Forecast)
                forecast_rows = df_arima[df_arima['Type'] == 'Forecast']
                
                if not forecast_rows.empty:
                    next_day_forecast = forecast_rows.iloc[0]['Close']
                    
                    # Tính toán chênh lệch
                    change_arima = next_day_forecast - current_price_arima
                    pct_change_arima = (change_arima / current_price_arima) * 100
                    
                    m4.metric(
                        label="Dự báo ngày mai",
                        value=f"{next_day_forecast:,.0f} VND",
                        delta=f"{change_arima:,.0f} VND ({pct_change_arima:.2f}%)"
                    )
                else:
                    m4.metric("Dự báo ngày mai", "N/A", "Chưa có dữ liệu")
            except Exception as e:
                m4.warning("Lỗi tính giá")

            st.divider()

            # --- PHẦN 2: BIỂU ĐỒ ---
            st.markdown("##### 📉 Biểu đồ Thực tế vs Dự báo")
            fig_arima = px.line(df_arima, x='Date', y='Close', color='Type',
                                title=f"Mô hình ARIMA - {target_bank} - Dự báo 30 ngày tới",
                                color_discrete_map={"History": "#A6CEE3", "Forecast": "#E31A1C"})
            
            # Cấu hình thao tác chuột (Pan/Zoom) & Ẩn toolbar
            fig_arima.update_layout(dragmode='pan')
            st.plotly_chart(fig_arima, use_container_width=True, 
                            config={'displayModeBar': False, 'scrollZoom': True})

        else:
            st.warning(f"⚠️ Chưa có dữ liệu ARIMA cho **{target_bank}**.")
            st.info(f"Vui lòng chạy file train ARIMA để tạo folder: `{arima_folder}`")


    # 2. SUB-TAB LSTM
    with sub_tab_lstm:
        lstm_folder = f"Save_model_LSTM/{target_bank}.VN"
        model_path = os.path.join(lstm_folder, "lstm.h5")
        scaler_path = os.path.join(lstm_folder, "lstm.pkl")
        loss_path = os.path.join(lstm_folder, "lstm.json") 
        old_loss_path = os.path.join(lstm_folder, "model_loss.json")
        result_csv_path = os.path.join(lstm_folder, "lstm.csv")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                # Load resources
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                
                st.markdown(f"##### 📊 Kết quả đánh giá LSTM - {target_bank}")
                m1, m2, m3, m4 = st.columns(4)

                # Metrics
                metrics = {}
                if os.path.exists(loss_path):
                    with open(loss_path, 'r') as f: metrics = json.load(f).get("LSTM", {})
                elif os.path.exists(old_loss_path):
                    with open(old_loss_path, 'r') as f: metrics = json.load(f).get("LSTM", {})
                
                m1.metric(
                    "R2 Score", 
                    f"{metrics.get('r2', 0):.4f}",
                    help="R-squared (Hệ số xác định).\n\nCàng gần 1 càng TỐT. Cho biết mô hình giải thích được bao nhiêu % sự biến thiên của dữ liệu."
                )
                m2.metric(
                    "RMSE", 
                    f"{metrics.get('rmse', 0):,.0f}",
                    help="Root Mean Squared Error.\n\nCàng THẤP càng tốt. Sai số trung bình phương gốc (đơn vị: VND)."
                )
                m3.metric(
                    "MAE", 
                    f"{metrics.get('mae', 0):,.0f}",
                    help="Mean Absolute Error.\n\nCàng THẤP càng tốt. Sai số tuyệt đối trung bình (đơn vị: VND)."
                )

                # Dự báo
                last_60_days = df[target_bank].values[-60:].reshape(-1, 1)
                last_60_scaled = scaler.transform(last_60_days)
                X_test = last_60_scaled.reshape(1, 60, 1)
                
                pred_scaled = model.predict(X_test)
                pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                
                change = pred_price - current_price
                pct_change = (change / current_price) * 100
                
                m4.metric(
                    label="Dự báo ngày mai",
                    value=f"{pred_price:,.0f} VND",
                    delta=f"{change:,.0f} VND ({pct_change:.2f}%)"
                )

                st.divider()

                # Biểu đồ
                if os.path.exists(result_csv_path):
                    df_res = pd.read_csv(result_csv_path)
                    st.markdown("##### 📉 Kiểm thử trên dữ liệu quá khứ (Test Set)")
                    df_melt = df_res.melt(id_vars='Date', value_vars=['Actual', 'Prediction'], 
                                          var_name='Legend', value_name='Price')
                    fig_lstm = px.line(df_melt, x='Date', y='Price', color='Legend',
                                       title=f"Mô hình LSTM - {target_bank}",
                                       color_discrete_map={"Actual": "#A6CEE3", "Prediction": "#E31A1C"},
                                       height=500)
                    
                    # --- CẤU HÌNH MỚI CHO LSTM ---
                    # 1. dragmode='pan': Kéo chuột là di chuyển
                    fig_lstm.update_layout(dragmode='pan')
                    
                    # 2. scrollZoom=True: Lăn chuột là Zoom
                    st.plotly_chart(fig_lstm, use_container_width=True, 
                                    config={'displayModeBar': False, 'scrollZoom': True})
                else:
                    st.warning("⚠️ Chưa có file kết quả test. Hãy Train lại model.")

            except Exception as e:
                st.error(f"Lỗi xử lý LSTM: {e}")
        else:
            st.warning(f"⚠️ Chưa tìm thấy model LSTM cho **{target_bank}**. Vui lòng sang tab 'Control Panel' để huấn luyện.")


# --- TAB 6: CONTROL PANEL ---
with tab6:
    st.header("Hệ thống Quản trị Dữ liệu & Mô hình")
    
    col1, col2 = st.columns(2)
    
    # --- CỘT 1: QUẢN LÝ DỮ LIỆU ---
    with col1:
        st.subheader("1. Cập nhật Dữ liệu")
        
        # 1. Chọn chế độ tải
        download_mode = st.radio(
            "Chọn phương thức tải:",
            options=["🔄 Cập nhật danh sách mặc định", "➕ Tải mã mới (Tùy chọn)"]
        )
        
        custom_tickers = []
        
        # 2. Nếu chọn tải mã mới thì hiện ô nhập liệu
        if download_mode == "➕ Tải mã mới (Tùy chọn)":
            raw_text = st.text_area(
                "Nhập mã cổ phiếu (phân cách bằng dấu phẩy hoặc khoảng trắng):",
                placeholder="Ví dụ: HPG, FPT, VNM..."
            )
            
            # Xử lý chuỗi nhập vào
            if raw_text:
                # Tách chuỗi, xóa khoảng trắng thừa, viết hoa
                # Hỗ trợ tách bằng cả dấu phẩy và khoảng trắng
                raw_list = raw_text.replace(',', ' ').split()
                
                for t in raw_list:
                    clean_t = t.strip().upper()
                    # Tự động thêm đuôi .VN nếu thiếu
                    if not clean_t.endswith(".VN"):
                        clean_t += ".VN"
                    custom_tickers.append(clean_t)
                
                if custom_tickers:
                    st.caption(f"Hệ thống sẽ tải: {', '.join(custom_tickers)}")

        # 3. Nút bấm thực thi
        if st.button("📥 Bắt đầu Tải Data", use_container_width=True):
            with st.spinner('Đang kết nối API để tải dữ liệu...'):
                try:
                    if download_mode == "➕ Tải mã mới (Tùy chọn)":
                        if not custom_tickers:
                            st.warning("Vui lòng nhập ít nhất 1 mã cổ phiếu.")
                            st.stop()
                        # Gọi hàm với danh sách tùy chọn
                        msg = dt_process.download_data(custom_list=custom_tickers)
                    else:
                        # Gọi hàm mặc định (custom_list=None)
                        msg = dt_process.download_data()
                    
                    st.success(msg)
                    load_data.clear()
                    time.sleep(1)
                    st.rerun() # Load lại trang để cập nhật danh sách bên Sidebar
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        # Nút Làm sạch Data (Giữ nguyên)
        if st.button("🧹 Làm sạch Dữ liệu (Clean)", use_container_width=True):
            with st.spinner('Đang xử lý Missing values & Outliers...'):
                msg = dt_process.clean_data()
                st.success(msg)

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