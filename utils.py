import pandas as pd
import streamlit as st
import os
from datetime import datetime
from typing import List, Tuple, Optional

# --- 1. 스타일링 관련 함수 ---

def load_css(file_path: str):
    """지정된 경로의 CSS 파일을 읽어 Streamlit 앱에 적용합니다."""
    if not os.path.exists(file_path):
        st.warning(f"CSS file not found at path: {file_path}")
        return
    # 파일을 열 때 encoding='utf-8' 옵션을 추가합니다.
    with open(file_path, "r", encoding="utf-8") as f: #
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# --- 2. 데이터 로딩 및 처리 함수 ---

# CSV 경로 설정
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'v_inspection_result_202509181119.csv')

@st.cache_data  # Streamlit 캐싱 (대용량 데이터 필수)
def load_data(date_col: str = 'ai_date_time') -> pd.DataFrame:
    """
    지정된 경로에서 CSV 데이터를 로드하고 기본 전처리를 수행합니다.
    - date_col: 날짜/시간 정보가 있는 컬럼명
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH, low_memory=False)
    
    # 날짜 컬럼을 datetime 객체로 변환
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        st.error(f"'{date_col}' column not found in the data.")
    
    print(f"Data loaded: {len(df):,} rows, {len(df.columns)} columns")  # 콘솔 로그
    return df

def apply_filters(
    df: pd.DataFrame, 
    date_range: Optional[Tuple[datetime.date, datetime.date]], 
    selected_equipment: List[str], 
    selected_customer: List[str], 
    itemcode: str, 
    lot: str
) -> pd.DataFrame:
    """데이터프레임에 다양한 필터를 순차적으로 적용합니다."""
    
    filtered_df = df.copy()
    
    # 1. 날짜 필터 (ai_date_time 컬럼 기준)
    if date_range and 'ai_date_time' in filtered_df.columns:
        start_date, end_date = date_range
        mask_date = (filtered_df['ai_date_time'].dt.date >= start_date) & \
                    (filtered_df['ai_date_time'].dt.date <= end_date)
        filtered_df = filtered_df[mask_date]
    
    # 2. 장비 필터 (inspection_machine 컬럼 기준)
    if selected_equipment:
        filtered_df = filtered_df[filtered_df['inspection_machine'].isin(selected_equipment)]
    
    # 3. 고객 필터 (rms_customer 컬럼 기준)
    if selected_customer:
        filtered_df = filtered_df[filtered_df['rms_customer'].isin(selected_customer)]
    
    # 4. ITEM CODE 필터 (barcode 컬럼 기준, 대소문자 무시)
    if itemcode and 'barcode' in filtered_df.columns:
        # 'barcode' 컬럼의 NaN 값을 빈 문자열로 대체 후 소문자로 변환하여 비교
        search_term = itemcode.strip().lower()
        filtered_df = filtered_df[
            filtered_df['barcode'].fillna('').str.lower().str.contains(search_term, na=False)
        ]
    
    # 5. LOT 필터 (lot_no 컬럼 기준, 대소문자 무시)
    if lot and 'lot_no' in filtered_df.columns:
        # 'lot_no' 컬럼의 NaN 값을 빈 문자열로 대체 후 소문자로 변환하여 비교
        search_term = lot.strip().lower()
        filtered_df = filtered_df[
            filtered_df['lot_no'].fillna('').str.lower().str.contains(search_term, na=False)
        ]
    
    return filtered_df