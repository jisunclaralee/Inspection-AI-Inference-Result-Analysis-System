import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Tuple, Optional
from sqlalchemy import create_engine
import plotly.express as px
import gc  # Garbage collector 추가
import numpy as np  # numpy import 추가
from pandas.api.types import is_numeric_dtype  # 추가: 숫자 타입 확인을 위한 import

# --- 1. 데이터 로딩 및 처리 함수 (메모리 최적화) ---
@st.cache_resource
def init_connection():
    """데이터베이스 연결 엔진을 생성하고 캐시에 저장합니다."""
    try:
        db_info = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_info['user']}:{db_info['password']}"
            f"@{db_info['host']}:{db_info['port']}/{db_info['dbname']}"
        )
        return create_engine(db_uri)
    except Exception as e:
        st.error(f"데이터베이스 연결 정보를 확인해주세요 (.streamlit/secrets.toml): {e}")
        return None

@st.cache_data(ttl=3600)  # 1시간 캐시 TTL 설정
def load_data(date_col: str = 'ai_date_time') -> pd.DataFrame:
    """PostgreSQL 데이터베이스에서 전체 데이터를 한번만 로드하고 캐시에 저장합니다."""
    conn = init_connection()
    if conn is None: 
        return pd.DataFrame()
    
    sql_query = "SELECT * FROM public.inspection_data;"
    try:
        # 메모리 사용량을 줄이기 위한 chunked reading
        chunksize = 50000  # 50,000 rows씩 읽기
        chunks = []
        
        for chunk in pd.read_sql(sql_query, conn, chunksize=chunksize):
            # 필요한 컬럼만 선택하여 메모리 절약
            if 'ai_date_time' in chunk.columns:
                chunk['ai_date_time'] = pd.to_datetime(chunk['ai_date_time'], errors='coerce')
            if 'body_created_at' in chunk.columns:
                chunk['body_created_at'] = pd.to_datetime(chunk['body_created_at'], errors='coerce')
            if 'header_created_at' in chunk.columns:
                chunk['header_created_at'] = pd.to_datetime(chunk['header_created_at'], errors='coerce')
            
            # 데이터 타입 최적화
            optimize_dataframe(chunk)
            chunks.append(chunk)
            print(f"Loaded chunk: {len(chunk):,} rows")
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            print(f"Data loaded from DB: {len(df):,} rows, {len(df.columns)} columns")
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"데이터베이스 로딩 중 오류 발생: {e}")
        return pd.DataFrame()

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임의 메모리 사용량을 최적화합니다."""
    for col in df.columns:
        col_type = df[col].dtype
        
        # 수치형 컬럼 최적화 (datetime 타입 제외)
        if is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(c_min) != 'nan' and str(c_max) != 'nan':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        
        # 카테고리컬 컬럼 최적화 (object 타입에 대해)
        elif col_type == object and len(df[col].unique()) / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
    
    return df

def apply_filters_efficient(
    df: pd.DataFrame,
    date_range: Optional[Tuple[datetime.date, datetime.date]],
    selected_afvi_equipment: List[str],
    selected_customer: List[str],
    selected_itemcode: Optional[str],
    selected_lot: Optional[str],
    selected_bundle: Optional[str]
) -> pd.DataFrame:
    """메모리 효율적으로 데이터프레임에 다양한 필터를 순차적으로 적용합니다."""
    # 깊은 복사 대신 원본 데이터 참조 (필요할 때만 복사)
    filtered_df = df
    
    # 날짜 필터링 (가장 효과적인 필터링 먼저)
    if date_range and 'ai_date_time' in filtered_df.columns:
        start_date, end_date = date_range
        valid_dates = filtered_df.dropna(subset=['ai_date_time'])
        mask_date = (valid_dates['ai_date_time'].dt.date >= start_date) & \
                    (valid_dates['ai_date_time'].dt.date <= end_date)
        filtered_df = valid_dates[mask_date].copy()  # 여기서만 복사
        
        # 메모리 정리
        del valid_dates
        gc.collect()
    
    # 장비 필터링
    if selected_afvi_equipment:
        mask_equipment = filtered_df['inspection_machine'].isin(selected_afvi_equipment)
        filtered_df = filtered_df[mask_equipment].copy()
        del mask_equipment
    
    # 고객 필터링
    if selected_customer:
        mask_customer = filtered_df['rms_customer'].isin(selected_customer)
        filtered_df = filtered_df[mask_customer].copy()
        del mask_customer
    
    # 아이템코드 필터링
    if selected_itemcode and selected_itemcode != "All":
        mask_item = filtered_df['barcode'] == selected_itemcode
        filtered_df = filtered_df[mask_item].copy()
        del mask_item
    
    # 로트 필터링
    if selected_lot and selected_lot != "All":
        mask_lot = filtered_df['lot_no'] == selected_lot
        filtered_df = filtered_df[mask_lot].copy()
        del mask_lot
    
    # Bundle 필터링
    if selected_bundle and selected_bundle != "All":
        if 'bundle_no' in filtered_df.columns:
            mask_bundle = filtered_df['bundle_no'] == selected_bundle
            filtered_df = filtered_df[mask_bundle].copy()
            del mask_bundle
        else:
            st.warning("Warning: 'bundle_no' column not found in data. Bundle filter is ignored.")
    
    return filtered_df

# --- 2. Streamlit 앱 메인 로직 ---

st.set_page_config(layout="wide")
st.title("Inspection AI inference result analysis system")

# 메모리 관리
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None

# 데이터 로딩 (메모리 모니터링 추가)
with st.spinner('Loading data from database...'):
    df_raw = load_data()

# 메모리 사용량 표시
if not df_raw.empty:
    memory_usage = df_raw.memory_usage(deep=True).sum() / 1024**2
    st.sidebar.info(f"📊 Data loaded: {len(df_raw):,} rows, {len(df_raw.columns)} columns\n💾 Memory: {memory_usage:.1f} MB")

if df_raw.empty:
    st.warning("데이터를 불러오지 못했습니다. 데이터베이스 연결 상태 또는 테이블을 확인해주세요.")
else:
    # --- 사이드바 필터 UI 구성 ---
    st.sidebar.header("Filter Options")
    date_min = df_raw['ai_date_time'].dropna().min().date()
    date_max = df_raw['ai_date_time'].dropna().max().date()
    date_range = st.sidebar.date_input("Date Range", (date_min, date_max), min_value=date_min, max_value=date_max)
    
    # Search 버튼
    if st.sidebar.button("Search", key="search_top"):
        with st.spinner('Applying filters... This may take a moment.'):
            st.session_state.filtered_data = apply_filters_efficient(
                df_raw, date_range,
                st.session_state.get('selected_afvi_equipment', []),
                st.session_state.get('selected_customer', []),
                st.session_state.get('selected_itemcode'),
                st.session_state.get('selected_lot'),
                st.session_state.get('selected_bundle')
            )
        # 메모리 정리
        gc.collect()
    
    st.sidebar.markdown("---")

    # 필터 위젯들 (메모리 사용량 표시 포함)
    equipment_counts = df_raw['inspection_machine'].value_counts()
    all_afvi_equipment = sorted(df_raw['inspection_machine'].dropna().unique())
    st.session_state.selected_afvi_equipment = st.sidebar.multiselect(
        "AFVI Equipment",
        all_afvi_equipment,
        format_func=lambda opt: f"{opt} ({equipment_counts.get(opt, 0)})"
    )

    customer_counts = df_raw['rms_customer'].value_counts()
    all_customers = sorted(df_raw['rms_customer'].dropna().unique())
    st.session_state.selected_customer = st.sidebar.multiselect(
        "Customer",
        all_customers,
        format_func=lambda opt: f"{opt} ({customer_counts.get(opt, 0)})"
    )

    # Bundle NO 필터
    if 'bundle_no' in df_raw.columns:
        bundle_counts = df_raw['bundle_no'].value_counts()
        bundle_options = ["All"] + sorted(df_raw['bundle_no'].dropna().unique().tolist())
        def format_bundle_with_count(opt):
            if opt == "All":
                return f"All ({len(df_raw)})"
            return f"{opt} ({bundle_counts.get(opt, 0)})"
        st.session_state.selected_bundle = st.sidebar.selectbox(
            "Bundle NO",
            bundle_options,
            format_func=format_bundle_with_count
        )
    else:
        st.session_state.selected_bundle = "All"

    lot_counts = df_raw['lot_no'].value_counts()
    lot_options = ["All"] + sorted(df_raw['lot_no'].dropna().unique().tolist())
    def format_lot_with_count(opt):
        if opt == "All":
            return f"All ({len(df_raw)})"
        return f"{opt} ({lot_counts.get(opt, 0)})"
    st.session_state.selected_lot = st.sidebar.selectbox(
        "Lot No",
        lot_options,
        format_func=format_lot_with_count
    )

    itemcode_counts = df_raw['barcode'].value_counts()
    item_options = ["All"] + sorted(df_raw['barcode'].dropna().unique().tolist())
    def format_itemcode_with_count(opt):
        if opt == "All":
            return f"All ({len(df_raw)})"
        return f"{opt} ({itemcode_counts.get(opt, 0)})"
    st.session_state.selected_itemcode = st.sidebar.selectbox(
        "Item Code",
        item_options,
        format_func=format_itemcode_with_count
    )
    
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Search", key="search_bottom"):
            with st.spinner('Applying filters... This may take a moment.'):
                st.session_state.filtered_data = apply_filters_efficient(
                    df_raw, date_range,
                    st.session_state.selected_afvi_equipment,
                    st.session_state.selected_customer,
                    st.session_state.selected_itemcode,
                    st.session_state.selected_lot,
                    st.session_state.selected_bundle
                )
            gc.collect()
    with col2:
        if st.button("Stop"):
            st.stop()
            
    # --- 데이터 표시 ---
    if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
        df_filtered = st.session_state.filtered_data
        
        # 메모리 사용량 표시
        filtered_memory = df_filtered.memory_usage(deep=True).sum() / 1024**2
        st.info(f"🔍 Filtered data: {len(df_filtered):,} rows\n💾 Memory: {filtered_memory:.1f} MB")
        
        # 탭 생성
        tab1, tab2, tab3, tab4 = st.tabs(["LOT ANALYSIS", "STRIP MAP", "IMAGE VIEWER", "TREND ANALYSIS"])

        with tab1:
            st.header("Lot Analysis")
            
            # Lot Hierarchy 필터 추가
            defect_option = st.radio("LOT HIERARCHY", ["All", "DEFECT", "GOOD"], horizontal=True)
            
            # 메모리 효율적인 필터링
            lot_df = df_filtered.copy()
            if defect_option == "DEFECT":
                lot_df = lot_df[lot_df['has_defect'] == 1]
            elif defect_option == "GOOD":
                lot_df = lot_df[lot_df['has_defect'] == 0]

            # 메모리 절약을 위해 1000개만 표시
            display_df = lot_df.head(1000)
            st.dataframe(display_df)
            st.info(f"Showing {len(display_df):,} of {len(lot_df):,} records.")
            
            # 메모리 정리
            del lot_df
            gc.collect()

        with tab2:
            st.header("Strip Map")
            st.write("Displaying defect locations on a sample strip map.")
            
            # 메모리 효율적인 defect 데이터 필터링
            if 'has_defect' in df_filtered.columns:
                defect_map_df = df_filtered[df_filtered['has_defect'] == 1].dropna(subset=['rel_x_strip', 'rel_y_strip'])
            else:
                defect_map_df = df_filtered.dropna(subset=['rel_x_strip', 'rel_y_strip'])
                
            # 최대 1000개 포인트만 표시
            if len(defect_map_df) > 1000:
                defect_map_df = defect_map_df.sample(1000, random_state=42)
            
            if not defect_map_df.empty:
                fig = px.scatter(defect_map_df, x='rel_x_strip', y='rel_y_strip', 
                                 title=f"Defect Distribution Map ({len(defect_map_df)} points)",
                                 labels={'rel_x_strip': 'X-coordinate', 'rel_y_strip': 'Y-coordinate'},
                                 hover_data=['defect_code'])
                fig.update_layout(plot_bgcolor='black', xaxis_range=[0,1], yaxis_range=[0,1])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No defects found in the filtered data.")
            
            # 메모리 정리
            del defect_map_df
            gc.collect()

        with tab3:
            st.header("Image Viewer")
            st.write("Displaying defect images.")

            # 이미지 경로가 있는 데이터만 필터링 (최대 20개)
            image_df = df_filtered.dropna(subset=['image_path']).head(20)
            
            if not image_df.empty:
                # 4개의 컬럼으로 이미지 갤러리 생성
                cols = st.columns(4)
                for index, row in image_df.iterrows():
                    with cols[index % 4]:
                        try:
                            st.image(row['image_path'], caption=f"Defect: {row['defect_code']}", use_column_width=True)
                        except Exception as e:
                            st.error(f"Cannot load image: {e}")
            else:
                st.info("No images to display in the filtered data.")
            st.warning("NOTE: Images will only be displayed if the server can access the file paths (e.g., /AI/M32V2/...). This works on local machines but may not work on a deployed web server.")
            
            # 메모리 정리
            del image_df
            gc.collect()

        with tab4:
            st.header("Trend Analysis")
            st.write("Daily defect trend.")

            # 메모리 효율적인 트렌드 분석
            trend_df = df_filtered[['ai_date_time', 'has_defect']].copy()
            if not trend_df.empty and 'ai_date_time' in trend_df.columns:
                trend_df['date'] = trend_df['ai_date_time'].dt.date
                daily_defect_counts = trend_df[trend_df['has_defect'] == 1].groupby('date').size().reset_index(name='defect_count')
                
                if not daily_defect_counts.empty:
                    fig = px.bar(daily_defect_counts, x='date', y='defect_count', 
                                title="Daily Defect Trend")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No defects found for trend analysis.")
            else:
                st.warning("No data for trend analysis.")
            
            # 메모리 정리
            del trend_df, daily_defect_counts
            gc.collect()
    else:
        st.info("Please set filters in the sidebar and click 'Search' to see the data.")

# 페이지 하단에 메모리 정리
gc.collect()