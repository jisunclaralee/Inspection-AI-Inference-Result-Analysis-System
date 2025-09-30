import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Tuple, Optional
from sqlalchemy import create_engine
import plotly.express as px
import gc
import numpy as np
from pandas.api.types import is_numeric_dtype
# from streamlit_plotly_events import plotly_events # 현재 UI에서는 불필요하므로 주석 처리

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

# [변경] Item Code 추출 함수: 앞뒤에 '/'를 포함하도록 수정
def extract_item_code(barcode_path: str) -> str:
    if not isinstance(barcode_path, str):
        return str(barcode_path)
    
    parts = barcode_path.split('/')
    if len(parts) >= 3:
        return f"/{parts[2]}/"
    return barcode_path

def optimize_dataframe(df: pd.DataFrame):
    # ... (기존 메모리 최적화 코드는 변경 없음)
    for col in df.columns:
        col_type = df[col].dtype
        if is_numeric_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if pd.notna(c_min) and pd.notna(c_max):
                if col_type.kind == 'i' or (col_type.kind == 'f' and df[col].mod(1).eq(0).all()):
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        elif col_type == 'object' and len(df[col].unique()) / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
    return df

@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    # ... (기존 데이터 로딩 코드는 변경 없음)
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

# [변경] apply_filters 함수: 다중 선택(List)을 처리하도록 수정
def apply_filters(
    df: pd.DataFrame,
    date_range: Optional[Tuple[datetime.date, datetime.date]],
    selected_afvi_equipment: List[str],
    selected_customer: List[str],
    selected_itemcodes: List[str], # List로 변경
    selected_lots: List[str],       # List로 변경
    selected_bundles: List[str]   # List로 변경
) -> pd.DataFrame:
    filtered_df = df
    if date_range and 'ai_date_time' in filtered_df.columns:
        mask_date = (filtered_df['ai_date_time'].dt.date >= date_range[0]) & (filtered_df['ai_date_time'].dt.date <= date_range[1])
        filtered_df = filtered_df.loc[mask_date]
    
    # 다중 선택이므로 .isin()을 사용하여 필터링
    if selected_afvi_equipment:
        filtered_df = filtered_df[filtered_df['inspection_machine'].isin(selected_afvi_equipment)]
    if selected_customer:
        filtered_df = filtered_df[filtered_df['rms_customer'].isin(selected_customer)]
    if selected_itemcodes:
        filtered_df = filtered_df[filtered_df['display_item_code'].isin(selected_itemcodes)]
    if selected_lots:
        filtered_df = filtered_df[filtered_df['lot_no'].isin(selected_lots)]
    
    bundle_col_name = 'bundle_no'
    if selected_bundles:
        if bundle_col_name in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[bundle_col_name].isin(selected_bundles)]
            
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
    date_range = st.sidebar.date_input(
        "Date Range", 
        (df_raw['ai_date_time'].dropna().min().date(), df_raw['ai_date_time'].dropna().max().date())
    )
    
    def run_search():
        with st.spinner('Applying filters...'):
            st.session_state.filtered_data = apply_filters(
                df_raw, date_range,
                st.session_state.selected_afvi_equipment, st.session_state.selected_customer,
                st.session_state.selected_itemcode, st.session_state.selected_lot,
                st.session_state.selected_bundle
            )
            gc.collect()

    # [변경] 모든 필터를 다중 선택(multiselect)으로 변경
    equipment_counts = df_raw['inspection_machine'].value_counts()
    st.session_state.selected_afvi_equipment = st.sidebar.multiselect(
        "AFVI Equipment", equipment_counts.index, format_func=lambda opt: f"{opt} ({equipment_counts[opt]})"
    )
    customer_counts = df_raw['rms_customer'].value_counts()
    st.session_state.selected_customer = st.sidebar.multiselect(
        "Customer", customer_counts.index, format_func=lambda opt: f"{opt} ({customer_counts[opt]})"
    )
    lot_counts = df_raw['lot_no'].value_counts()
    st.session_state.selected_lot = st.sidebar.multiselect(
        "Lot No", lot_counts.index, format_func=lambda opt: f"{opt} ({lot_counts[opt]})"
    )
    bundle_col_name = 'bundle_no'
    if bundle_col_name in df_raw.columns:
        bundle_counts = df_raw[bundle_col_name].value_counts()
        st.session_state.selected_bundle = st.sidebar.multiselect(
            "Bundle NO", bundle_counts.index, format_func=lambda opt: f"{opt} ({bundle_counts[opt]})"
        )
    else: 
        st.session_state.selected_bundle = []
        
    itemcode_counts = df_raw['display_item_code'].value_counts()
    st.session_state.selected_itemcode = st.sidebar.multiselect(
        "Item Code", itemcode_counts.index, format_func=lambda opt: f"{opt} ({itemcode_counts[opt]})"
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

        # (기존 코드의 with tab1: 부분을 아래 코드로 대체)
       
        with tab1:
            # --- 상태 변수 초기화 ---
            if 'view_level' not in st.session_state:
                st.session_state.view_level = 'lot_summary'
            if 'selected_test_id' not in st.session_state:
                st.session_state.selected_test_id = None
            if 'selected_bundle_info' not in st.session_state:
                st.session_state.selected_bundle_info = None
            if 'selected_strip_id' not in st.session_state: # [추가] Strip 선택 상태
                st.session_state.selected_strip_id = None

            # --- 뷰(View) 렌더링 함수 정의 ---

            # Level 1: Lot 요약 뷰 (기존과 동일)
            def display_lot_summary_view(df):
                st.header("Lot Summary")
                st.caption("`test_id`를 클릭하여 해당 Lot의 Bundle 리스트를 확인하세요.")
                if 'test_id' not in df.columns:
                    st.error("'test_id' 컬럼을 찾을 수 없습니다.")
                    return
                lot_summary = df.groupby(['test_id', 'lot_no', 'rms_customer']).size().reset_index(name='image_count')
                for _, row in lot_summary.iterrows():
                    cols = st.columns([0.2, 0.3, 0.3, 0.2])
                    if cols[0].button(str(row['test_id']), key=f"test_id_{row['test_id']}", use_container_width=True):
                        st.session_state.selected_test_id = row['test_id']
                        st.session_state.view_level = 'bundle_summary'
                        st.rerun()
                    cols[1].info(f"**Lot:** {row['lot_no']}")
                    cols[2].info(f"**Customer:** {row['rms_customer']}")
                    cols[3].info(f"**Images:** {row['image_count']:,}")
                    st.divider()

            # Level 2: Bundle 요약 뷰 (기존과 동일)
            def display_bundle_summary_view(df, test_id):
                if st.button("⬅️ Lot 요약으로 돌아가기"):
                    st.session_state.view_level = 'lot_summary'
                    st.session_state.selected_test_id = None
                    st.rerun()
                st.markdown(f"## Bundle Summary for Test ID: <span style='color:#66FFCC;'>`{test_id}`</span>", unsafe_allow_html=True)
                st.caption("`bundle_no`를 클릭하여 해당 Bundle의 Strip 리스트를 확인하세요.")
                bundle_df = df[df['test_id'] == test_id]
                bundle_summary = bundle_df.groupby(['test_id', 'lot_no', 'bundle_no']).size().reset_index(name='image_count')
                for _, row in bundle_summary.iterrows():
                    cols = st.columns([0.4, 0.4, 0.2])
                    if cols[0].button(row['bundle_no'], key=f"bundle_{row['bundle_no']}", use_container_width=True):
                        st.session_state.selected_bundle_info = row.to_dict()
                        st.session_state.view_level = 'strip_summary'
                        st.rerun()
                    cols[1].info(f"**Lot:** {row['lot_no']}")
                    cols[2].info(f"**Images:** {row['image_count']:,}")
                    st.divider()
            
            # [변경] Level 3: Strip 요약 뷰
            def display_strip_summary_view(df, bundle_info):
                if st.button("⬅️ Bundle 요약으로 돌아가기"):
                    st.session_state.view_level = 'bundle_summary'
                    st.session_state.selected_bundle_info = None
                    st.rerun()

                st.markdown(f"## Strip Summary for Bundle: <span style='color:#66FFCC;'>`{bundle_info['bundle_no']}`</span>", unsafe_allow_html=True)
                st.caption("`strip_id`를 클릭하여 상세 이미지를 확인하세요.")
                
                strip_df = df[(df['test_id'] == bundle_info['test_id']) & (df['lot_no'] == bundle_info['lot_no']) & (df['bundle_no'] == bundle_info['bundle_no'])]
                defect_col_name = 'afvi_ai_defect'
                if 'strip_id' in strip_df.columns and defect_col_name in strip_df.columns:
                    strip_summary = strip_df.groupby('strip_id').agg(
                        defect_count=('afvi_ai_defect', 'nunique'),
                        image_count=('strip_id', 'size')
                    ).reset_index()

                    for _, row in strip_summary.iterrows():
                        cols = st.columns([0.4, 0.4, 0.2])
                        if cols[0].button(row['strip_id'], key=f"strip_{row['strip_id']}", use_container_width=True):
                            st.session_state.selected_strip_id = row['strip_id']
                            st.session_state.view_level = 'image_detail' # 다음 뷰로 전환
                            st.rerun()
                        cols[1].info(f"**Defect Types:** {row['defect_count']}")
                        cols[2].info(f"**Images:** {row['image_count']:,}")
                        st.divider()
                else:
                    st.warning(f"'strip_id' 또는 '{defect_col_name}' 컬럼을 찾을 수 없습니다.")

            # [추가] Level 4: 이미지 상세 뷰
            def display_image_detail_view(df, bundle_info, strip_id):
                if st.button("⬅️ Strip 요약으로 돌아가기"):
                    st.session_state.view_level = 'strip_summary'
                    st.session_state.selected_strip_id = None
                    st.rerun()

                st.markdown(f"### Image Details for Strip: `{strip_id}`")
                
                image_df = df[
                    (df['test_id'] == bundle_info['test_id']) &
                    (df['lot_no'] == bundle_info['lot_no']) &
                    (df['bundle_no'] == bundle_info['bundle_no']) &
                    (df['strip_id'] == strip_id)
                ]

                if image_df.empty:
                    st.info("표시할 이미지가 없습니다.")
                else:
                    # 각 이미지의 경로와 이미지를 함께 표시
                    for _, row in image_df.iterrows():
                        cols = st.columns([0.7, 0.3])
                        with cols[0]:
                            st.text(row['image_path']) # 이미지 경로 주소 표시
                            st.caption(f"Defect: {row.get('afvi_ai_defect', 'N/A')}")
                        with cols[1]:
                            # 이미지 경로가 유효하면 이미지를, 아니면 에러 아이콘을 보여줌
                            st.image(row['image_path'], use_column_width=True)
                        st.divider()

            # --- 뷰 라우터(Router) ---
            if st.session_state.view_level == 'lot_summary':
                display_lot_summary_view(df_filtered)
            elif st.session_state.view_level == 'bundle_summary':
                display_bundle_summary_view(df_filtered, st.session_state.selected_test_id)
            elif st.session_state.view_level == 'strip_summary':
                display_strip_summary_view(df_filtered, st.session_state.selected_bundle_info)
            elif st.session_state.view_level == 'image_detail':
                display_image_detail_view(df_filtered, st.session_state.selected_bundle_info, st.session_state.selected_strip_id)

            # --- 3. CSS 스타일 주입 및 뷰 라우터 ---
            
            # [추가] 버튼과 정보 박스 스타일 통일을 위한 CSS
            st.markdown("""
            <style>
                /* 정보 박스 스타일 */
                .info-box {
                    background-color: #0E1117; /* Streamlit 기본 다크모드 배경색 */
                    border: 1px solid #31333F;
                    border-radius: 0.5rem;
                    padding: 10px;
                    height: 50px; /* 고정 높이 */
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                }
                /* 버튼 스타일 (높이 맞추기) */
                .stButton > button {
                    height: 50px; /* 정보 박스와 동일한 높이 */
                }
            </style>
            """, unsafe_allow_html=True)

            # st.session_state.view_level 값에 따라 적절한 뷰를 렌더링합니다.
            if st.session_state.view_level == 'lot_summary':
                display_lot_summary_view(df_filtered)
            elif st.session_state.view_level == 'bundle_summary':
                display_bundle_summary_view(df_filtered, st.session_state.selected_test_id)
            elif st.session_state.view_level == 'strip_summary':
                display_strip_summary_view(df_filtered, st.session_state.selected_bundle_info)
        
        # ... (tab2, tab3, tab4 코드는 이전과 유사하게 유지)

    else:
        st.info("Please set filters in the sidebar and click 'Search' to see the data.")