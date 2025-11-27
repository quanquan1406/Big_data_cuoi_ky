import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------------------------
# 1. CẤU HÌNH TRANG DASHBOARD
# -----------------------------------------------------------
st.set_page_config(layout="wide", page_title="Bank Stock Analysis Dashboard")

st.title("📈 Dashboard Phân Tích Cổ Phiếu Ngân Hàng (2020-2024)")
st.markdown("Đề tài: Phân tích biến động giá và Quản trị rủi ro nhóm ngân hàng trên HOSE.")

# -----------------------------------------------------------
# 2. HÀM TẢI DỮ LIỆU (CACHE ĐỂ CHẠY NHANH)
# -----------------------------------------------------------
@st.cache_data
def load_data():
    folder_path = 'data' # Đảm bảo thư mục data nằm cùng chỗ với file này
    df_merged = pd.DataFrame()
    
    if not os.path.exists(folder_path):
        return None
        
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            df_temp = pd.read_csv(file_path)
            # Xử lý ngày tháng
            col_date = 'Date' if 'Date' in df_temp.columns else 'Ngay'
            df_temp[col_date] = pd.to_datetime(df_temp[col_date])
            df_temp.set_index(col_date, inplace=True)
            
            # Lấy tên mã
            ticker = file.split('.')[0].replace('.VN','')
            
            # Lấy giá điều chỉnh
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

# Gọi hàm tải dữ liệu
df = load_data()

if df is None:
    st.error("Không tìm thấy thư mục 'data'. Vui lòng kiểm tra lại.")
    st.stop()

# -----------------------------------------------------------
# 3. SIDEBAR (THANH ĐIỀU KHIỂN BÊN TRÁI)
# -----------------------------------------------------------
st.sidebar.header("Bộ lọc dữ liệu")

# Chọn ngân hàng
all_banks = df.columns.tolist()
selected_banks = st.sidebar.multiselect("Chọn ngân hàng để so sánh:", all_banks, default=all_banks[:4])

# Chọn khoảng thời gian
start_date = df.index.min()
end_date = df.index.max()
date_range = st.sidebar.date_input("Chọn khoảng thời gian:", [start_date, end_date])

# Lọc dữ liệu theo lựa chọn
if not selected_banks:
    st.warning("Vui lòng chọn ít nhất 1 ngân hàng.")
    st.stop()

df_filtered = df[selected_banks]
df_filtered = df_filtered[(df_filtered.index >= pd.to_datetime(date_range[0])) & 
                          (df_filtered.index <= pd.to_datetime(date_range[1]))]

# Tính toán các chỉ số cơ bản
df_returns = df_filtered.pct_change().dropna()

# -----------------------------------------------------------
# 4. GIAO DIỆN CHÍNH (TABS)
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Hiệu suất & Xu hướng", "⚠️ Phân tích Rủi ro", "🎯 Tương quan & Danh mục"])

with tab1:
    st.header("Biến động giá và Tăng trưởng")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Biến động giá hàng ngày")
        st.line_chart(df_filtered) # Biểu đồ tương tác mặc định của Streamlit
        
    with col2:
        st.subheader("Tăng trưởng tích lũy (Cumulative Return)")
        # Tính tăng trưởng: (Giá sau / Giá đầu) - 1
        cumulative_ret = (df_filtered / df_filtered.iloc[0]) - 1
        st.line_chart(cumulative_ret)
        
    st.metric(label="Số ngày giao dịch", value=len(df_filtered))

with tab2:
    st.header("Đánh giá mức độ Rủi ro")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân phối lợi nhuận (Boxplot)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df_returns * 100, ax=ax)
        ax.set_ylabel("Lợi nhuận (%)")
        st.pyplot(fig) # Hiển thị biểu đồ matplotlib lên web
        
    with col2:
        st.subheader("Sụt giảm tối đa (Drawdown)")
        rolling_max = df_filtered.cummax()
        drawdown = df_filtered / rolling_max - 1.0
        st.line_chart(drawdown)

    st.subheader("Biến động lịch sử (30-Day Rolling Volatility)")
    volatility = df_returns.rolling(window=30).std() * (252**0.5)
    st.line_chart(volatility)

with tab3:
    st.header("Tương quan và Hiệu quả đầu tư")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ma trận tương quan (Correlation)")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)
        
    with col2:
        st.subheader("Rủi ro vs Lợi nhuận (Risk-Return)")
        rets = df_returns.mean() * 252
        risk = df_returns.std() * (252 ** 0.5)
        
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
        ax_scatter.scatter(risk, rets, s=200, c='teal', alpha=0.6)
        
        for label, x, y in zip(rets.index, risk, rets):
            ax_scatter.annotate(label, xy=(x, y), xytext=(0, 0), 
                                textcoords='offset points', ha='center', va='center', color='black', weight='bold')
            
        ax_scatter.set_xlabel("Rủi ro (Volatility)")
        ax_scatter.set_ylabel("Lợi nhuận (Return)")
        ax_scatter.grid(True, linestyle='--')
        st.pyplot(fig_scatter)