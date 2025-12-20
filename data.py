import yfinance as yf
import pandas as pd
import os
import datetime as dt
import glob
from pathlib import Path

# --- CẤU HÌNH ---
BANK_STOCKS = ['VCB.VN', 'BID.VN', 'CTG.VN', 'TCB.VN', 'VPB.VN', 
               'MBB.VN', 'ACB.VN', 'HDB.VN', 'LPB.VN', 'STB.VN']
DATA_DIR_NAME = 'data'

def get_data_dir():
    """Hàm phụ trợ lấy đường dẫn tuyệt đối đến folder data"""
    base_directory = os.getcwd()
    data_directory = os.path.join(base_directory, DATA_DIR_NAME)
    os.makedirs(data_directory, exist_ok=True)
    return data_directory



def download_data(custom_list=None):
    """
    Tải dữ liệu chứng khoán.
    Args:
        custom_list (list): Danh sách mã tùy chọn (VD: ['HPG.VN', 'FPT.VN']). 
                            Nếu None thì tải danh sách mặc định BANK_STOCKS.
    """
    try:
        data_directory = get_data_dir()
        start_date = dt.datetime.today() - dt.timedelta(5 * 365)
        end_date = dt.datetime.today()
        
        # LOGIC MỚI: Chọn danh sách để tải
        if custom_list and len(custom_list) > 0:
            stocks_to_download = custom_list
            mode_msg = "danh sách tùy chọn"
        else:
            stocks_to_download = BANK_STOCKS
            mode_msg = "danh sách mặc định (Ngân hàng)"
            
        count_success = 0
        print(f"⬇️ Bắt đầu tải {mode_msg} vào: {data_directory}")
        
        for stock in stocks_to_download:
            try:
                # Tải data
                data = yf.download(stock, start=start_date, end=end_date, progress=False)
                
                if not data.empty:
                    data.reset_index(inplace=True)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)

                    csv_file_path = os.path.join(data_directory, f'{stock}.csv')
                    data.to_csv(csv_file_path, index=False)
                    count_success += 1
            except Exception as e:
                print(f"Lỗi tải mã {stock}: {e}")
                continue

        return f"✅ Đã tải thành công {count_success} mã ({mode_msg})."
    
    except Exception as e:
        return f"❌ Lỗi nghiêm trọng: {str(e)}"

def _clean_single_csv(csv_path, overwrite=True):
    """
    Hàm nội bộ: Làm sạch 1 file CSV cụ thể (Logic của bạn).
    """
    csv_path = Path(csv_path)
    out_dir = csv_path.parent if overwrite else (csv_path.parent / 'cleaned')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Đọc dữ liệu
    df = pd.read_csv(csv_path)

    # 2. Xử lý cột Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        # Cố gắng reset index nếu date nằm ở index cũ
        try:
            df = df.reset_index()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except:
            pass

    # Loại bỏ hàng không có ngày
    if 'Date' in df.columns:
        df = df[~df['Date'].isna()].copy()

    # 3. Loại bỏ cột rác (Unnamed)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', regex=True)]

    # 4. Loại bỏ duplicate và sort
    if 'Date' in df.columns:
        df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
    else:
        df = df.drop_duplicates().reset_index(drop=True)

    # 5. Chuyển đổi số học (Numeric)
    numeric_cols = [c for c in df.columns if c != 'Date' and c != 'Ticker']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 6. Xử lý Missing Value cho các cột giá
    price_cols = [c for c in ['Open','High','Low','Close','Adj Close'] if c in df.columns]
    if price_cols:
        # Bỏ hàng nát (toàn NaN)
        df = df[~df[price_cols].isna().all(axis=1)].copy()
        # Fill lỗ hổng dữ liệu (Forward fill trước, rồi Backfill)
        df[price_cols] = df[price_cols].ffill().bfill()

    # 7. Xử lý Volume
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0).astype('int64')

    # 8. Lưu file
    out_path = Path(out_dir) / csv_path.name
    df.to_csv(out_path, index=False)
    
    return len(df)

def clean_data():
    """
    Hàm quét toàn bộ thư mục data và làm sạch tất cả file .csv tìm thấy.
    Trả về thông báo trạng thái (str).
    """
    try:
        data_directory = get_data_dir()
        csv_files = glob.glob(os.path.join(data_directory, '*.csv'))
        
        if not csv_files:
            return "⚠️ Không tìm thấy file CSV nào để làm sạch."

        count_cleaned = 0
        total_rows = 0
        
        for f in csv_files:
            try:
                rows = _clean_single_csv(f, overwrite=True)
                total_rows += rows
                count_cleaned += 1
            except Exception as e:
                print(f"Lỗi khi clean file {f}: {e}")
        
        return f"🧹 Đã làm sạch {count_cleaned} file. Tổng cộng {total_rows} dòng dữ liệu sẵn sàng."

    except Exception as e:
        return f"❌ Lỗi quá trình làm sạch: {str(e)}"

# --- ĐOẠN NÀY ĐỂ TEST KHI CHẠY TRỰC TIẾP FILE NÀY ---
if __name__ == "__main__":
    print("--- Test Download ---")
    print(download_data())
    print("\n--- Test Clean ---")
    print(clean_data())