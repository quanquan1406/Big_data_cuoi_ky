# Import thư viện cần thiết
import os
import glob
from pathlib import Path
import pandas as pd

def clean_stock_csv(csv_path, out_dir=None, overwrite=True):
    """
    Đọc file CSV đã tải từ yfinance, làm sạch và lưu.
    Nếu `overwrite=True` thì ghi đè file gốc trong cùng thư mục `data/`.
    Nếu `overwrite=False` thì lưu vào `out_dir` (mặc định là `data/cleaned/`).
    Trả về: (đường_dẫn_đã_lưu, dataframe)
    """
    csv_path = Path(csv_path)

    if overwrite:
        out_dir = csv_path.parent
    else:
        out_dir = Path(out_dir) if out_dir is not None else (csv_path.parent / 'cleaned')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu (không ép parse Date để linh hoạt)
    df = pd.read_csv(csv_path)

    # Nếu có cột 'Date', parse thành datetime; nếu không, cố gắng lấy từ index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        # Một số file có thể đã lưu Date làm index
        try:
            df = df.reset_index()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception:
            pass

    # Loại bỏ hàng không có ngày hợp lệ
    if 'Date' in df.columns:
        df = df[~df['Date'].isna()].copy()

    # Loại bỏ cột Unnamed nếu có (lỗi khi save CSV đôi khi tạo cột này)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', regex=True)]

    # Loại bỏ duplicate theo Date (giữ bản ghi đầu tiên) và sắp xếp
    if 'Date' in df.columns:
        df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
    else:
        df = df.drop_duplicates().reset_index(drop=True)

    # Chuyển đổi các cột (trừ Date) sang số nếu có thể
    numeric_cols = [c for c in df.columns if c != 'Date']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Xác định các cột giá thường có trong dữ liệu yfinance
    price_cols = [c for c in ['Open','High','Low','Close','Adj Close'] if c in df.columns]
    if price_cols:
        # Loại bỏ hàng mà tất cả các cột giá đều NaN
        df = df[~df[price_cols].isna().all(axis=1)].copy()
        # Điền missing bằng forward-fill rồi back-fill để lấp các lỗ nhỏ
        df[price_cols] = df[price_cols].ffill().bfill()

    # Xử lý Volume nếu tồn tại (fill 0 và cast int)
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0).astype('int64')


    # Lưu file đã làm sạch (ghi đè nếu overwrite=True)
    out_path = Path(out_dir) / csv_path.name
    df.to_csv(out_path, index=False)

    return out_path, df

# Thực thi làm sạch cho tất cả file CSV trong `data/` và tạo file tổng hợp (ghi đè trực tiếp)
base_dir = Path.cwd()
data_dir = base_dir / 'data'
data_dir.mkdir(parents=True, exist_ok=True)
csv_files = sorted(glob.glob(str(data_dir / '*.csv')))

cleaned_paths = []
dfs = []
for f in csv_files:
    try:
        # Ghi đè trực tiếp file trong thư mục data/
        out_path, df_clean = clean_stock_csv(f, overwrite=True)
        print(f'Đã lưu file đã làm sạch (ghi đè): {out_path} (hàng={len(df_clean)})')
        cleaned_paths.append(out_path)
        # Gắn tên ticker từ tên file (ví dụ 'VCB.VN.csv' -> 'VCB.VN')
        ticker = Path(f).stem
        df_with_ticker = df_clean.copy()
        df_with_ticker['Ticker'] = ticker
        dfs.append(df_with_ticker)
    except Exception as e:
        print(f'Lỗi khi xử lý {f}: {e}')
