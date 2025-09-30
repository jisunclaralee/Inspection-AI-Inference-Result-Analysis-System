import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Tuple, Optional
from sqlalchemy import create_engine
import gc
import numpy as np
from pandas.api.types import is_numeric_dtype

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

def extract_item_code(file_path: str) -> str:
    """Item Code를 경로에서 추출합니다 (예: .../CODE/... -> CODE)"""
    if not isinstance(file_path, str):
        return str(file_path)
    
    parts = file_path.split('/')
    if len(parts) >= 3:
        return parts[2]
    return file_path

def optimize_dataframe(df: pd.DataFrame):
    """데이터프레임의 메모리 사용량을 최적화합니다."""
    # (기존 코드는 변경 없음)
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
    """PostgreSQL에서 데이터를 청크 단위로 로드하고 최적화합니다."""
    # (기존 코드는 변경 없음)
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
    selected_itemcodes: List[str],
    selected_lots: List[str],
    selected_bundles: List[str]
) -> pd.DataFrame:
    """세션 상태의 필터 값을 기반으로 데이터프레임을 필터링합니다."""
    # (기존 코드는 변경 없음)
    filtered_df = df.copy() # 원본 수정을 막기 위해 복사본 사용

    if date_range and len(date_range) == 2 and 'ai_date_time' in filtered_df.columns:
        mask_date = (filtered_df['ai_date_time'].dt.date >= date_range[0]) & (filtered_df['ai_date_time'].dt.date <= date_range[1])
        filtered_df = filtered_df.loc[mask_date]
    
    if selected_afvi_equipment:
        filtered_df = filtered_df[filtered_df['inspection_machine'].isin(selected_afvi_equipment)]
    if selected_customer:
        filtered_df = filtered_df[filtered_df['rms_customer'].isin(selected_customer)]
    if selected_itemcodes:
        filtered_df = filtered_df[filtered_df['display_item_code'].isin(selected_itemcodes)]
    if selected_lots:
        filtered_df = filtered_df[filtered_df['lot_no'].isin(selected_lots)]
    
    bundle_col_name = 'bundle_no'
    if selected_bundles and bundle_col_name in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[bundle_col_name].isin(selected_bundles)]
            
    return filtered_df

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
    # Item Code 생성 (file_name 컬럼 사용, 실제 컬럼명에 맞게 수정 필요)
    if 'display_item_code' not in df_raw.columns and 'file_name' in df_raw.columns:
        df_raw['display_item_code'] = df_raw['file_name'].apply(extract_item_code).astype('category')
    elif 'display_item_code' not in df_raw.columns:
         st.error("'display_item_code' 또는 'file_name' 컬럼을 찾을 수 없어 Item Code를 생성할 수 없습니다.")


    memory_usage = df_raw.memory_usage(deep=True).sum() / 1024**2
    st.sidebar.info(f" Data loaded: {len(df_raw):,} rows\n Memory: {memory_usage:.1f} MB")

    # 🔽🔽🔽 --- 사이드바 필터 UI (연쇄 필터 적용) --- 🔽🔽🔽
    st.sidebar.header("Filter Options")
    
    options_df = df_raw.copy()

    # 1. Date Range 필터
    min_date = options_df['ai_date_time'].dropna().min().date()
    max_date = options_df['ai_date_time'].dropna().max().date()
    st.sidebar.date_input("Date Range", (min_date, max_date), key='date_range_filter')
    
    if st.session_state.date_range_filter and len(st.session_state.date_range_filter) == 2:
        options_df = options_df[
            (options_df['ai_date_time'].dt.date >= st.session_state.date_range_filter[0]) &
            (options_df['ai_date_time'].dt.date <= st.session_state.date_range_filter[1])
        ]

    # 2. Equipment 필터
    equipment_options = options_df['inspection_machine'].value_counts()
    st.sidebar.multiselect(
        "AFVI Equipment", options=equipment_options.index, key='selected_afvi_equipment',
        format_func=lambda opt: f"{opt} ({equipment_options.get(opt, 0)})"
    )
    if st.session_state.selected_afvi_equipment:
        options_df = options_df[options_df['inspection_machine'].isin(st.session_state.selected_afvi_equipment)]

    # 3. Customer 필터
    customer_options = options_df['rms_customer'].value_counts()
    is_customer_disabled = len(customer_options) == 0
    st.sidebar.multiselect(
        "Customer", options=customer_options.index, key='selected_customer',
        format_func=lambda opt: f"{opt} ({customer_options.get(opt, 0)})",
        disabled=is_customer_disabled,
        help="선택 가능한 고객사 데이터가 없습니다." if is_customer_disabled else "고객사를 선택하세요."
    )
    if st.session_state.selected_customer:
        options_df = options_df[options_df['rms_customer'].isin(st.session_state.selected_customer)]
        
    # 4. Lot No 필터
    lot_options = options_df['lot_no'].value_counts()
    is_lot_disabled = len(lot_options) == 0
    st.sidebar.multiselect(
        "Lot No", options=lot_options.index, key='selected_lot',
        format_func=lambda opt: f"{opt} ({lot_options.get(opt, 0)})",
        disabled=is_lot_disabled,
        help="선택 가능한 Lot 데이터가 없습니다." if is_lot_disabled else "Lot No를 선택하세요."
    )
    if st.session_state.selected_lot:
        options_df = options_df[options_df['lot_no'].isin(st.session_state.selected_lot)]
        
    # 5. Bundle NO 필터
    bundle_col_name = 'bundle_no'
    if bundle_col_name in options_df.columns:
        bundle_options = options_df[bundle_col_name].value_counts()
        is_bundle_disabled = len(bundle_options) == 0
        st.sidebar.multiselect(
            "Bundle NO", options=bundle_options.index, key='selected_bundle',
            format_func=lambda opt: f"{opt} ({bundle_options.get(opt, 0)})",
            disabled=is_bundle_disabled,
            help="선택 가능한 Bundle 데이터가 없습니다." if is_bundle_disabled else "Bundle No를 선택하세요."
        )
    else:
        st.session_state.selected_bundle = []

    # 6. Item Code 필터
    if 'display_item_code' in options_df.columns:
        itemcode_options = options_df['display_item_code'].value_counts()
        is_item_disabled = len(itemcode_options) == 0
        st.sidebar.multiselect(
            "Item Code", options=itemcode_options.index, key='selected_itemcode',
            format_func=lambda opt: f"{opt} ({itemcode_options.get(opt, 0)})",
            disabled=is_item_disabled,
            help="선택 가능한 Item Code 데이터가 없습니다." if is_item_disabled else "Item Code를 선택하세요."
        )

    def run_search():
        with st.spinner('Applying filters...'):
            st.session_state.view_level = 'lot_summary' # 검색 시 항상 첫 화면으로 리셋

            st.session_state.filtered_data = apply_filters(
                df_raw,
                st.session_state.date_range_filter,
                st.session_state.selected_afvi_equipment,
                st.session_state.selected_customer,
                st.session_state.get('selected_itemcode', []),
                st.session_state.selected_lot,
                st.session_state.get('selected_bundle', [])
            )
            gc.collect()

    st.sidebar.markdown("---")
    if st.sidebar.button("Search", use_container_width=True):
        run_search()
    if st.sidebar.button("Stop", use_container_width=True):
        st.stop()
    # 🔼🔼🔼 --- 사이드바 필터 UI 종료 --- 🔼🔼🔼

    # --- 데이터 표시 ---
    if st.session_state.filtered_data is None:
        st.info("Please set filters in the sidebar and click 'Search' to see the data.")
    else:
        df_filtered = st.session_state.filtered_data
        
        filtered_memory = df_filtered.memory_usage(deep=True).sum() / 1024**2
        st.info(f" Filtered data: {len(df_filtered):,} rows | Memory: {filtered_memory:.1f} MB")
        
        tab1, tab2, tab3, tab4 = st.tabs(["LOT ANALYSIS", "STRIP MAP", "IMAGE VIEWER", "TREND ANALYSIS"])

        with tab1:
            # --- 1. 상태 변수 초기화 ---
            if 'view_level' not in st.session_state: st.session_state.view_level = 'lot_summary'
            if 'selected_test_id' not in st.session_state: st.session_state.selected_test_id = None
            if 'selected_bundle_info' not in st.session_state: st.session_state.selected_bundle_info = None
            if 'selected_strip_info' not in st.session_state: st.session_state.selected_strip_info = None

            # --- 2. CSS 스타일 주입 ---
            st.markdown("""
            <style>
                .info-box { background-color: #262730; border: 1px solid #31333F; border-radius: 0.5rem; padding: 10px 15px; height: 60px; display: flex; flex-direction: column; justify-content: center; }
                .stButton > button { height: 60px; }
            </style>
            """, unsafe_allow_html=True)

            # --- 3. 뷰(View) 렌더링 함수 정의 (기존 코드와 동일) ---

            # Level 1: Lot 요약 뷰
            def display_lot_summary_view(df):
                st.header("Lot Summary")
                st.caption("`Test ID`를 클릭하여 해당 Lot의 Bundle 리스트를 확인하세요.")
                if 'test_id' not in df.columns:
                    st.error("'test_id' 컬럼을 찾을 수 없습니다.")
                    return
                
                header_cols = st.columns([0.2, 0.4, 0.2, 0.2])
                header_cols[0].markdown("**Test ID**")
                header_cols[1].markdown("**Lot No & Customer**")
                header_cols[2].markdown("**Image Count**")
                st.divider()

                lot_summary = df.groupby(['test_id', 'lot_no', 'rms_customer']).size().reset_index(name='image_count')
                for _, row in lot_summary.iterrows():
                    unique_key = f"test_id_{row['test_id']}_{row['lot_no']}"
                    cols = st.columns([0.2, 0.4, 0.2, 0.2])
                    if cols[0].button(str(row['test_id']), key=unique_key, use_container_width=True):
                        st.session_state.selected_test_id = row['test_id']
                        st.session_state.view_level = 'bundle_summary'
                        st.rerun()
                    cols[1].markdown(f"<div class='info-box' style='align-items: flex-start;'><div><strong>Lot:</strong> {row['lot_no']}</div><small>Customer: {row['rms_customer']}</small></div>", unsafe_allow_html=True)
                    cols[2].markdown(f"<div class='info-box' style='align-items: center;'>{row['image_count']:,}</div>", unsafe_allow_html=True)

            # Level 2: Bundle 요약 뷰
            def display_bundle_summary_view(df, test_id):
                if st.button("◀️ Lot Summary"):
                    st.session_state.view_level = 'lot_summary'
                    st.session_state.selected_test_id = None
                    st.rerun()
                st.markdown(f"## Bundle Summary for Test ID: <span style='color:#66FFCC;'>`{test_id}`</span>", unsafe_allow_html=True)
                st.caption("`Bundle No`를 클릭하여 해당 Bundle의 Strip 리스트를 확인하세요.")
                
                header_cols = st.columns([0.4, 0.4, 0.2])
                header_cols[0].markdown("**Bundle No**")
                header_cols[1].markdown("**Lot No**")
                header_cols[2].markdown("**Image Count**")
                st.divider()

                bundle_df = df[df['test_id'] == test_id]
                bundle_summary = bundle_df.groupby(['test_id', 'lot_no', 'bundle_no']).size().reset_index(name='image_count')
                for _, row in bundle_summary.iterrows():
                    unique_key = f"bundle_{row['test_id']}_{row['bundle_no']}"
                    cols = st.columns([0.4, 0.4, 0.2])
                    if cols[0].button(row['bundle_no'], key=unique_key, use_container_width=True):
                        st.session_state.selected_bundle_info = row.to_dict()
                        st.session_state.view_level = 'strip_summary'
                        st.rerun()
                    cols[1].markdown(f"<div class='info-box' style='align-items: center;'>{row['lot_no']}</div>", unsafe_allow_html=True)
                    cols[2].markdown(f"<div class='info-box' style='align-items: center;'>{row['image_count']:,}</div>", unsafe_allow_html=True)

            # Level 3: Strip 요약 뷰
            def display_strip_summary_view(df, bundle_info):
                if st.button("◀️ Bundle Summary"):
                    st.session_state.view_level = 'bundle_summary'
                    st.session_state.selected_bundle_info = None
                    st.rerun()
                st.markdown(f"## Strip Summary for Bundle: <span style='color:#66FFCC;'>`{bundle_info['bundle_no']}`</span>", unsafe_allow_html=True)
                st.caption("`Strip ID`를 기준으로 Defect 목록을 확인하고 '이미지 보기'를 클릭하세요.")
                st.divider()
                
                strip_df = df[(df['test_id'] == bundle_info['test_id']) & (df['lot_no'] == bundle_info['lot_no']) & (df['bundle_no'] == bundle_info['bundle_no'])]
                defect_col_name = 'afvi_ai_defect'

                if 'strip_id' not in strip_df.columns or defect_col_name not in strip_df.columns:
                    st.warning(f"'strip_id' 또는 '{defect_col_name}' 컬럼을 찾을 수 없습니다.")
                    return

                strip_summary = strip_df.groupby(['strip_id', defect_col_name]).size().reset_index(name='image_count')
                if strip_summary.empty:
                    st.info("해당 Bundle에는 분석할 Strip 데이터가 없습니다.")
                    return

                for strip_id, group in strip_summary.groupby('strip_id'):
                    st.markdown(f"#### Strip ID: `{strip_id}`")
                    header_cols = st.columns([0.6, 0.2, 0.2])
                    header_cols[0].markdown("**Defect Type**")
                    header_cols[1].markdown("**Image Count**")
                    
                    for _, row in group.iterrows():
                        unique_key = f"view_{strip_id}_{row[defect_col_name]}"
                        row_cols = st.columns([0.6, 0.2, 0.2])
                        row_cols[0].markdown(f"<div class='info-box' style='height: 50px; justify-content: center;'>{row[defect_col_name]}</div>", unsafe_allow_html=True)
                        row_cols[1].markdown(f"<div class='info-box' style='height: 50px; justify-content: center;'>{row['image_count']:,}</div>", unsafe_allow_html=True)
                        if row_cols[2].button("이미지 보기", key=unique_key, use_container_width=True, help=f"{strip_id}의 {row[defect_col_name]} 이미지 보기"):
                            full_strip_info = bundle_info.copy()
                            full_strip_info.update(row.to_dict())
                            st.session_state.selected_strip_info = full_strip_info
                            st.session_state.view_level = 'image_detail'
                            st.rerun()
                    st.divider()

            # Level 4: 이미지 상세 뷰
            def display_image_detail_view(df, strip_info):
                if st.button("◀️ Strip Summary"):
                    st.session_state.view_level = 'strip_summary'
                    st.session_state.selected_strip_info = None
                    st.rerun()
                
                st.markdown(f"### Image Paths for Strip: `{strip_info['strip_id']}`")
                st.caption(f"Defect Type: `{strip_info['afvi_ai_defect']}`")
                
                image_df = df[
                    (df['test_id'] == strip_info['test_id']) &
                    (df['lot_no'] == strip_info['lot_no']) &
                    (df['bundle_no'] == strip_info['bundle_no']) &
                    (df['strip_id'] == strip_info['strip_id']) &
                    (df['afvi_ai_defect'] == strip_info['afvi_ai_defect'])
                ].dropna(subset=['image_path'])

                if image_df.empty:
                    st.info("표시할 이미지가 없습니다.")
                else:
                    for path in image_df['image_path']:
                        st.code(path)

            # --- 4. 뷰 라우터(Router) ---
            if st.session_state.view_level == 'lot_summary':
                display_lot_summary_view(df_filtered)
            elif st.session_state.view_level == 'bundle_summary':
                display_bundle_summary_view(df_filtered, st.session_state.selected_test_id)
            elif st.session_state.view_level == 'strip_summary':
                display_strip_summary_view(df_filtered, st.session_state.selected_bundle_info)
            elif st.session_state.view_level == 'image_detail':
                display_image_detail_view(df_filtered, st.session_state.selected_strip_info)
                
        with tab2: st.header("Strip Map")
        with tab3: st.header("Image Viewer")
        with tab4: st.header("Trend Analysis")