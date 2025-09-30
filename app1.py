import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Tuple, Optional
from sqlalchemy import create_engine
import plotly.express as px
import gc
import numpy as np
from pandas.api.types import is_numeric_dtype
from streamlit_plotly_events import plotly_events # 반응형 차트 라이브러리

# --- 1. 데이터 로딩 및 처리 함수 ---

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
        st.error(f"데이터베이스 연결 엔진 생성 실패: {e}")
        return None

# [수정] Item Code 추출 함수: 슬래시 없이 순수한 코드만 반환
def extract_item_code(barcode_path: str) -> str:
    if not isinstance(barcode_path, str):
        return str(barcode_path)
    
    parts = barcode_path.split('/')
    if len(parts) >= 3:
        return parts[2] # 예: /M32V2/MCP22539B00-006/... 에서 MCP22539B00-006 추출
    return barcode_path

def optimize_dataframe(df: pd.DataFrame):
    """데이터프레임의 메모리 사용량을 최적화합니다."""
    for col in df.columns:
        col_type = df[col].dtype
        if is_numeric_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if pd.notna(c_min) and pd.notna(c_max):
                if col_type.kind == 'i' or (col_type.kind == 'f' and df[col].mod(1).eq(0).all()): # 정수형이거나 소수점 없는 실수형
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else: # 실수형
                    if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        elif col_type == 'object' and len(df[col].unique()) / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
    return df

@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """PostgreSQL에서 데이터를 청크 단위로 로드하고 최적화합니다."""
    engine = init_connection()
    if engine is None: return pd.DataFrame()
    
    sql_query = "SELECT * FROM public.inspection_data;"
    try:
        chunksize = 50000
        chunks = []
        for chunk in pd.read_sql(sql_query, engine, chunksize=chunksize):
            for col in ['ai_date_time', 'body_created_at', 'header_created_at']:
                if col in chunk.columns:
                    chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
            optimize_dataframe(chunk)
            chunks.append(chunk)
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            gc.collect()
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"데이터베이스 로딩 중 오류 발생: {e}")
        return pd.DataFrame()

def apply_filters(
    df: pd.DataFrame,
    date_range: Optional[Tuple[datetime.date, datetime.date]],
    selected_afvi_equipment: List[str],
    selected_customer: List[str],
    selected_itemcode: Optional[str],
    selected_lot: Optional[str],
    selected_bundle: Optional[str]
) -> pd.DataFrame:
    filtered_df = df
    if date_range and 'ai_date_time' in filtered_df.columns:
        mask_date = (filtered_df['ai_date_time'].dt.date >= date_range[0]) & (filtered_df['ai_date_time'].dt.date <= date_range[1])
        filtered_df = filtered_df.loc[mask_date]
    if selected_afvi_equipment:
        filtered_df = filtered_df[filtered_df['inspection_machine'].isin(selected_afvi_equipment)]
    if selected_customer:
        filtered_df = filtered_df[filtered_df['rms_customer'].isin(selected_customer)]
    if selected_itemcode and selected_itemcode != "All":
        filtered_df = filtered_df[filtered_df['display_item_code'] == selected_itemcode]
    if selected_lot and selected_lot != "All":
        filtered_df = filtered_df[filtered_df['lot_no'] == selected_lot]
    bundle_col_name = 'bundle_no'
    if selected_bundle and selected_bundle != "All":
        if bundle_col_name in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[bundle_col_name] == selected_bundle]
    return filtered_df.copy()

# --- 2. Streamlit 앱 메인 로직 ---
st.set_page_config(layout="wide")
st.title("Inspection AI inference result analysis system")

if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None

with st.spinner('Loading and optimizing data...'):
    df_raw = load_data()

if df_raw.empty:
    st.warning("데이터를 불러오지 못했습니다. DB 연결을 확인해주세요.")
else:
    if 'display_item_code' not in df_raw.columns:
        df_raw['display_item_code'] = df_raw['barcode'].apply(extract_item_code).astype('category')

    memory_usage = df_raw.memory_usage(deep=True).sum() / 1024**2
    st.sidebar.info(f"📊 Data loaded: {len(df_raw):,} rows\n💾 Memory: {memory_usage:.1f} MB")

    # --- 사이드바 필터 UI ---
    st.sidebar.header("Filter Options")
    date_min = df_raw['ai_date_time'].dropna().min().date()
    date_max = df_raw['ai_date_time'].dropna().max().date()
    date_range = st.sidebar.date_input("Date Range", (date_min, date_max), min_value=date_min, max_value=date_max)
    
    def run_search():
        with st.spinner('Applying filters...'):
            st.session_state.filtered_data = apply_filters(
                df_raw, date_range,
                st.session_state.selected_afvi_equipment, st.session_state.selected_customer,
                st.session_state.selected_itemcode, st.session_state.selected_lot,
                st.session_state.selected_bundle
            )
            gc.collect()

    equipment_counts = df_raw['inspection_machine'].value_counts()
    st.session_state.selected_afvi_equipment = st.sidebar.multiselect(
        "AFVI Equipment", equipment_counts.index, format_func=lambda opt: f"{opt} ({equipment_counts[opt]})"
    )
    customer_counts = df_raw['rms_customer'].value_counts()
    st.session_state.selected_customer = st.sidebar.multiselect(
        "Customer", customer_counts.index, format_func=lambda opt: f"{opt} ({customer_counts[opt]})"
    )
    lot_counts = df_raw['lot_no'].value_counts()
    lot_options = ["All"] + lot_counts.index.tolist()
    st.session_state.selected_lot = st.sidebar.selectbox(
        "Lot No", lot_options, format_func=lambda opt: f"All ({len(df_raw)})" if opt == "All" else f"{opt} ({lot_counts[opt]})"
    )
    bundle_col_name = 'bundle_no'
    if bundle_col_name in df_raw.columns:
        bundle_counts = df_raw[bundle_col_name].value_counts()
        bundle_options = ["All"] + bundle_counts.index.tolist()
        st.session_state.selected_bundle = st.sidebar.selectbox(
            "Bundle NO", bundle_options, format_func=lambda opt: f"All ({len(df_raw)})" if opt == "All" else f"{opt} ({bundle_counts[opt]})"
        )
    else: st.session_state.selected_bundle = "All"
    itemcode_counts = df_raw['display_item_code'].value_counts()
    item_options = ["All"] + itemcode_counts.index.tolist()
    st.session_state.selected_itemcode = st.sidebar.selectbox(
        "Item Code", item_options, format_func=lambda opt: f"All ({len(df_raw)})" if opt == "All" else f"{opt} ({itemcode_counts[opt]})"
    )
    
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Search"):
            run_search()
    with col2:
        if st.button("Stop"):
            st.stop()

    # --- 데이터 표시 ---
    if st.session_state.filtered_data is not None:
        df_filtered = st.session_state.filtered_data
        
        filtered_memory = df_filtered.memory_usage(deep=True).sum() / 1024**2
        st.info(f"🔍 Filtered data: {len(df_filtered):,} rows | 💾 Memory: {filtered_memory:.1f} MB")
        
        tab1, tab2, tab3, tab4 = st.tabs(["LOT ANALYSIS", "STRIP MAP", "IMAGE VIEWER", "TREND ANALYSIS"])

        with tab1:
            st.header("Interactive Lot Analysis")
            if 'selected_lot_analysis' not in st.session_state: st.session_state.selected_lot_analysis = None
            if 'selected_bundle_analysis' not in st.session_state: st.session_state.selected_bundle_analysis = None

            if st.button("Reset View / Drill Up", use_container_width=True):
                st.session_state.selected_lot_analysis = None
                st.session_state.selected_bundle_analysis = None
                st.rerun()

            st.subheader("Lot Summary")
            lot_summary = df_filtered.groupby('lot_no').agg(image_count=('lot_no', 'size'), defect_count=('has_defect', 'sum')).reset_index()
            lot_summary['defect_rate'] = (lot_summary['defect_count'] / lot_summary['image_count']) * 100
            fig_lot = px.bar(lot_summary, x='image_count', y='lot_no', color='defect_rate', orientation='h', title='Lot별 이미지 수 및 불량률')
            fig_lot.update_layout(yaxis={'categoryorder':'total ascending'})
            selected_lot_points = plotly_events(fig_lot, click_event=True, key="lot_chart")
            
            if selected_lot_points:
                selected_lot = selected_lot_points[0]['y']
                if st.session_state.selected_lot_analysis != selected_lot:
                    st.session_state.selected_lot_analysis = selected_lot
                    st.session_state.selected_bundle_analysis = None
                    st.rerun()

            if st.session_state.selected_lot_analysis:
                st.markdown("---")
                st.subheader(f"Bundle Summary for Lot: `{st.session_state.selected_lot_analysis}`")
                lot_data = df_filtered[df_filtered['lot_no'] == st.session_state.selected_lot_analysis]
                bundle_summary = lot_data.groupby('bundle_no').agg(image_count=('bundle_no', 'size'), defect_count=('has_defect', 'sum')).reset_index()
                bundle_summary['defect_rate'] = (bundle_summary['defect_count'] / bundle_summary['image_count']) * 100
                fig_bundle = px.bar(bundle_summary, x='image_count', y='bundle_no', color='defect_rate', orientation='h', title='Bundle별 이미지 수 및 불량률')
                fig_bundle.update_layout(yaxis={'categoryorder':'total ascending'})
                selected_bundle_points = plotly_events(fig_bundle, click_event=True, key="bundle_chart")
                
                if selected_bundle_points:
                    st.session_state.selected_bundle_analysis = selected_bundle_points[0]['y']
                    st.rerun()

            if st.session_state.selected_bundle_analysis:
                st.markdown("---")
                st.subheader(f"Strip Details for Bundle: `{st.session_state.selected_bundle_analysis}`")
                bundle_data = df_filtered[(df_filtered['lot_no'] == st.session_state.selected_lot_analysis) & (df_filtered['bundle_no'] == st.session_state.selected_bundle_analysis)]
                if 'strip_id' in bundle_data.columns and 'defect_code' in bundle_data.columns:
                    strip_summary = bundle_data.groupby(['strip_id', 'defect_code', 'has_defect']).size().reset_index(name='image_count')
                    st.dataframe(strip_summary, use_container_width=True)
                else:
                    st.warning("'strip_id' 또는 'defect_code' 컬럼을 찾을 수 없습니다.")
        
        # ... (tab2, tab3, tab4)

    else:
        st.info("Please set filters in the sidebar and click 'Search' to see the data.")