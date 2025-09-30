import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Tuple, Optional
from sqlalchemy import create_engine
import gc
import os
import numpy as np
import plotly.express as px
from pandas.api.types import is_numeric_dtype
from streamlit_plotly_events import plotly_events

# --- 1. 데이터 로딩 및 처리 함수 ---

@st.cache_resource
def init_connection():
    """데이터베이스 연결 엔진을 생성하고 캐시에 저장합니다."""
    try:
        db_info = st.secrets["database"]
        db_uri = (f"postgresql+psycopg2://{db_info['user']}:{db_info['password']}"
                  f"@{db_info['host']}:{db_info['port']}/{db_info['dbname']}")
        return create_engine(db_uri)
    except Exception as e:
        st.error(f"데이터베이스 연결 엔진 생성 실패: {e}")
        return None

def extract_item_code(file_path: str) -> str:
    """Item Code를 경로에서 추출합니다 (예: .../CODE/... -> CODE)"""
    if not isinstance(file_path, str): return str(file_path)
    parts = file_path.split('/')
    if len(parts) >= 3: return parts[2]
    return file_path

def optimize_dataframe(df: pd.DataFrame):
    """데이터프레임의 메모리 사용량을 최적화합니다."""
    for col in df.columns:
        col_type = df[col].dtype
        if is_numeric_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if pd.notna(c_min) and pd.notna(c_max):
                if col_type.kind in 'iu' or (col_type.kind == 'f' and (df[col] == df[col].round()).all()):
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                elif col_type.kind == 'f':
                    if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
        elif col_type == 'object' and df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    return df

@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """PostgreSQL에서 데이터를 청크 단위로 로드하고 최적화합니다."""
    engine = init_connection()
    if engine is None: return pd.DataFrame()
    sql_query = "SELECT * FROM public.inspection_data;"
    try:
        chunks = [optimize_dataframe(chunk) for chunk in pd.read_sql(sql_query, engine, chunksize=50000, parse_dates=['ai_date_time', 'body_created_at', 'header_created_at'])]
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            gc.collect()
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"데이터베이스 로딩 중 오류 발생: {e}")
        return pd.DataFrame()

def apply_filters(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """세션 상태의 필터 값을 기반으로 데이터프레임을 필터링합니다."""
    filtered_df = df
    date_range = filters.get('date_range')
    if date_range and len(date_range) == 2 and 'ai_date_time' in filtered_df.columns:
        mask_date = (filtered_df['ai_date_time'].dt.date >= date_range[0]) & (filtered_df['ai_date_time'].dt.date <= date_range[1])
        filtered_df = filtered_df.loc[mask_date]
    
    col_map = {
        'selected_afvi_equipment': 'inspection_machine', 'selected_customer': 'rms_customer',
        'selected_itemcode': 'display_item_code', 'selected_lot': 'lot_no', 'selected_bundle': 'bundle_no'
    }
    for key, values in filters.items():
        if key.startswith('selected_') and values:
            db_col = col_map.get(key)
            if db_col and db_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[db_col].isin(values)]
    return filtered_df.copy()

def display_main_kpis(df: pd.DataFrame):
    """메인 화면 상단에 KPI를 표시합니다."""
    if df.empty:
        st.warning("선택된 필터에 해당하는 데이터가 없습니다.")
        return
    total, equip, cust = len(df), df['inspection_machine'].nunique(), df['rms_customer'].nunique()
    defect_rate = (df['has_defect'].sum() / total) * 100 if total > 0 else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 검사 수", f"{total:,}"); c2.metric("장비 수", f"{equip:,}")
    c3.metric("고객사 수", f"{cust:,}"); c4.metric("평균 불량률", f"{defect_rate:.2f}%")

# --- 3. 드릴다운 뷰(View) 함수 정의 ---
def display_lot_view(df):
    st.header("Lot List")
    st.caption("`Test ID`를 클릭하여 상세 분석을 시작하세요.")
    if 'test_id' not in df.columns: st.error("'test_id' 컬럼을 찾을 수 없습니다."); return
    header_cols = st.columns([0.2, 0.4, 0.2, 0.2]); header_cols[0].markdown("**Test ID**"); header_cols[1].markdown("**Lot No & Customer**"); header_cols[2].markdown("**Image Count**"); st.divider()
    lot_summary = df.groupby(['test_id', 'lot_no', 'rms_customer'], observed=False).size().reset_index(name='image_count')
    for _, row in lot_summary.iterrows():
        cols = st.columns([0.2, 0.4, 0.2, 0.2])
        if cols[0].button(str(row['test_id']), key=f"test_id_{row['test_id']}_{row['lot_no']}", use_container_width=True):
            st.session_state.selected_test_id = row['test_id']; st.session_state.view_level = 'bundle'; st.rerun()
        cols[1].markdown(f"<div class='info-box' style='align-items: flex-start;'><div><strong>Lot:</strong> {row['lot_no']}</div><small>Customer: {row['rms_customer']}</small></div>", unsafe_allow_html=True)
        cols[2].markdown(f"<div class='info-box' style='align-items: center;'>{row['image_count']:,}</div>", unsafe_allow_html=True)

def display_bundle_view(df, test_id):
    if st.button("◀ Lot List"): st.session_state.view_level = 'lot'; st.session_state.selected_test_id = None; st.rerun()
    st.markdown(f"## Bundle List for Test ID: <span style='color:#04F79A;'>`{test_id}`</span>", unsafe_allow_html=True)
    st.caption("`Bundle No`를 클릭하세요.")
    header_cols = st.columns([0.4, 0.4, 0.2]); header_cols[0].markdown("**Bundle No**"); header_cols[1].markdown("**Lot No**"); header_cols[2].markdown("**Image Count**"); st.divider()
    bundle_df = df[df['test_id'] == test_id]
    bundle_summary = bundle_df.groupby(['test_id', 'lot_no', 'bundle_no'], observed=False).size().reset_index(name='image_count')
    for _, row in bundle_summary.iterrows():
        cols = st.columns([0.4, 0.4, 0.2])
        if cols[0].button(row['bundle_no'], key=f"bundle_{row['test_id']}_{row['bundle_no']}", use_container_width=True):
            st.session_state.selected_bundle_info = row.to_dict(); st.session_state.view_level = 'strip'; st.rerun()
        cols[1].markdown(f"<div class='info-box' style='align-items: center;'>{row['lot_no']}</div>", unsafe_allow_html=True)
        cols[2].markdown(f"<div class='info-box' style='align-items: center;'>{row['image_count']:,}</div>", unsafe_allow_html=True)

def display_strip_view(df, bundle_info):
    if st.button("◀ Bundle List"): st.session_state.view_level = 'bundle'; st.session_state.selected_bundle_info = None; st.rerun()
    st.markdown(f"## Strip List for Bundle: <span style='color:#04F79A;'>`{bundle_info['bundle_no']}`</span>", unsafe_allow_html=True)
    st.caption("`Strip ID`를 클릭하여 Defect 통계를 확인하세요.")
    strip_df = df[(df['test_id'] == bundle_info['test_id']) & (df['lot_no'] == bundle_info['lot_no']) & (df['bundle_no'] == bundle_info['bundle_no'])]
    defect_col_name = 'afvi_ai_defect'
    if 'strip_id' in strip_df.columns:
        strip_summary = strip_df.groupby('strip_id', observed=False).agg(image_count=('strip_id', 'size'), defect_types=(defect_col_name, 'nunique')).reset_index()
        header_cols = st.columns([0.4, 0.4, 0.2]); header_cols[0].markdown("**Strip ID**"); header_cols[1].markdown("**Defect Type Count**"); header_cols[2].markdown("**Total Image Count**"); st.divider()
        for _, row in strip_summary.iterrows():
            cols = st.columns([0.4, 0.4, 0.2])
            if cols[0].button(row['strip_id'], key=f"strip_{bundle_info['bundle_no']}_{row['strip_id']}", use_container_width=True):
                st.session_state.selected_strip_id = row['strip_id']; st.session_state.view_level = 'defect'; st.rerun()
            cols[1].markdown(f"<div class='info-box' style='align-items: center;'>{row['defect_types']}</div>", unsafe_allow_html=True)
            cols[2].markdown(f"<div class='info-box' style='align-items: center;'>{row['image_count']:,}</div>", unsafe_allow_html=True)
    else: st.warning("'strip_id' 컬럼을 찾을 수 없습니다.")

def display_defect_view(df, bundle_info, strip_id):
    if st.button("◀ Strip List"): st.session_state.view_level = 'strip'; st.session_state.selected_strip_id = None; st.rerun()
    st.markdown(f"### Defect List for Strip: <span style='color:#04F79A;'>`{strip_id}`</span>", unsafe_allow_html=True)
    defect_df = df[(df['test_id'] == bundle_info['test_id']) & (df['lot_no'] == bundle_info['lot_no']) & (df['bundle_no'] == bundle_info['bundle_no']) & (df['strip_id'] == strip_id)]
    defect_col_name = 'afvi_ai_defect'
    if defect_col_name in defect_df.columns:
        defect_summary = defect_df.groupby(defect_col_name, observed=False).size().reset_index(name='image_count').sort_values('image_count', ascending=False)
        if not defect_summary.empty:
            fig = px.bar(defect_summary, x='image_count', y=defect_col_name, orientation='h', title='Defect Statistics'); fig.update_layout(yaxis={'categoryorder':'total ascending'}); st.plotly_chart(fig, use_container_width=True)
        st.caption("'이미지 보기' 버튼을 클릭하여 이미지 팝업을 확인하세요.")
        header_cols = st.columns([0.6, 0.2, 0.2]); header_cols[0].markdown(f"**{defect_col_name}**"); header_cols[1].markdown("**Image Count**"); st.divider()
        for _, row in defect_summary.iterrows():
            cols = st.columns([0.6, 0.2, 0.2])
            cols[0].markdown(f"<div class='info-box' style='align-items: center;'>{row[defect_col_name]}</div>", unsafe_allow_html=True)
            cols[1].markdown(f"<div class='info-box' style='align-items: center;'>{row['image_count']:,}</div>", unsafe_allow_html=True)
            if cols[2].button("이미지 보기", key=f"defect_{strip_id}_{row[defect_col_name]}", use_container_width=True):
                st.session_state.selected_defect_info = {**bundle_info, **row.to_dict(), 'strip_id': strip_id}
                st.session_state.view_level = 'image_path'; st.rerun()
    else: st.warning(f"'{defect_col_name}' 컬럼을 찾을 수 없습니다.")

def display_image_path_view(df, defect_info):
    if st.button("◀ Defect List"): st.session_state.view_level = 'defect'; st.session_state.selected_defect_info = None; st.rerun()
    defect_col_name = 'afvi_ai_defect'; st.markdown(f"#### Image Paths for Strip `{defect_info['strip_id']}` > Defect `{defect_info[defect_col_name]}`")
    st.caption("'이미지 보기' 버튼을 클릭하여 로컬 이미지를 팝업으로 확인하세요."); st.divider()
    LOCAL_IMAGE_DIRECTORY = r"C:\Users\user\Desktop\새 폴더\venv\pcb_dashboard\data\images"
    image_df = df[(df['test_id'] == defect_info['test_id']) & (df['lot_no'] == defect_info['lot_no']) & (df['bundle_no'] == defect_info['bundle_no']) & (df['strip_id'] == defect_info['strip_id']) & (df[defect_col_name] == defect_info[defect_col_name])].dropna(subset=['image_path'])
    if image_df.empty: st.info("표시할 이미지가 없습니다.")
    else:
        for index, row in image_df.iterrows():
            db_path = row['image_path']
            cols = st.columns([0.8, 0.2])
            cols[0].code(db_path, language=None)
            if cols[1].button("이미지 보기", key=f"popup_{index}", use_container_width=True):
                @st.dialog("Local Image Preview")
                def image_popup(image_db_path):
                    cleaned_path = image_db_path[1:] if image_db_path.startswith('/') else image_db_path
                    local_filename = cleaned_path.replace('/', '')
                    full_local_path = os.path.join(LOCAL_IMAGE_DIRECTORY, local_filename)
                    if os.path.exists(full_local_path):
                        st.image(full_local_path, use_column_width=True, caption=local_filename)
                    else:
                        st.error(f"Local file not found:\n{full_local_path}")
                image_popup(db_path)
            st.divider()

# --- 4. Streamlit 앱 메인 로직 ---
st.set_page_config(layout="wide")
st.title("Inspection AI Inference Result Analysis System")

if 'filtered_data' not in st.session_state: st.session_state.filtered_data = None

with st.spinner('Loading and optimizing data...'):
    df_raw = load_data()

if df_raw.empty:
    st.warning("데이터를 불러오지 못했습니다. DB 연결을 확인해주세요.")
else:
    if 'display_item_code' not in df_raw.columns and 'file_name' in df_raw.columns:
        df_raw['display_item_code'] = df_raw['file_name'].apply(extract_item_code).astype('category')
    
    memory_usage = df_raw.memory_usage(deep=True).sum() / 1024**2
    st.sidebar.info(f" Data loaded: {len(df_raw):,} rows\n Memory: {memory_usage:.1f} MB")
    st.sidebar.header("Filter Options")
    
    # --- 사이드바 연쇄 필터 UI ---
    df_options = df_raw.copy()
    date_range = st.sidebar.date_input("Date Range", (df_options['ai_date_time'].dropna().min().date(), df_options['ai_date_time'].dropna().max().date()))
    if date_range and len(date_range) == 2:
        df_options = df_options.loc[(df_options['ai_date_time'].dt.date >= date_range[0]) & (df_options['ai_date_time'].dt.date <= date_range[1])]

    def create_multiselect(label, col_name, df_source, session_key):
        counts = df_source[col_name].value_counts()
        if session_key not in st.session_state:
            st.session_state[session_key] = []
        selected = st.sidebar.multiselect(label, counts.index, key=session_key, format_func=lambda opt: f"{opt} ({counts.get(opt, 0)})")
        return df_source[df_source[col_name].isin(selected)] if selected else df_source

    df_options = create_multiselect("AFVI Equipment", "inspection_machine", df_options, 'selected_afvi_equipment')
    df_options = create_multiselect("Customer", "rms_customer", df_options, 'selected_customer')
    df_options = create_multiselect("Lot No", "lot_no", df_options, 'selected_lot')
    if 'bundle_no' in df_options.columns:
        df_options = create_multiselect("Bundle NO", "bundle_no", df_options, 'selected_bundle')
    if 'display_item_code' in df_options.columns:
        create_multiselect("Item Code", "display_item_code", df_options, 'selected_itemcode')

    def run_search():
        with st.spinner('Applying filters...'):
            st.session_state.view_level = 'lot'
            st.session_state.selected_test_id, st.session_state.selected_bundle_info, st.session_state.selected_strip_id, st.session_state.selected_defect_info = None, None, None, None
            filter_args = { 'date_range': date_range, **{k: v for k, v in st.session_state.items() if k.startswith('selected_')}}
            st.session_state.filtered_data = apply_filters(df_raw, **filter_args)
            gc.collect()

    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Search", use_container_width=True): run_search()
    if col2.button("Stop", use_container_width=True): st.stop()

    if st.session_state.filtered_data is None:
        display_main_kpis(df_raw)
        st.info("Please set filters in the sidebar and click 'Search' to see the data.")
    else:
        df_filtered = st.session_state.filtered_data
        display_main_kpis(df_filtered)
        st.divider()
        
        tab1, tab2, tab3, tab4 = st.tabs(["LOT ANALYSIS", "STRIP MAP", "IMAGE VIEWER", "TREND ANALYSIS"])

        with tab1:
            if 'view_level' not in st.session_state: st.session_state.view_level = 'lot'
            if 'selected_test_id' not in st.session_state: st.session_state.selected_test_id = None
            if 'selected_bundle_info' not in st.session_state: st.session_state.selected_bundle_info = None
            if 'selected_strip_id' not in st.session_state: st.session_state.selected_strip_id = None
            if 'selected_defect_info' not in st.session_state: st.session_state.selected_defect_info = None

            st.markdown("""<style>.info-box { background-color: #262730; border: 1px solid #31333F; border-radius: 0.5rem; padding: 10px 15px; height: 60px; display: flex; flex-direction: column; justify-content: center; } .stButton > button { height: 60px; }</style>""", unsafe_allow_html=True)

            if st.session_state.view_level == 'lot': display_lot_view(df_filtered)
            elif st.session_state.view_level == 'bundle': display_bundle_view(df_filtered, st.session_state.selected_test_id)
            elif st.session_state.view_level == 'strip': display_strip_view(df_filtered, st.session_state.selected_bundle_info)
            elif st.session_state.view_level == 'defect': display_defect_view(df_filtered, st.session_state.selected_bundle_info, st.session_state.selected_strip_id)
            elif st.session_state.view_level == 'image_path': display_image_path_view(df_filtered, st.session_state.selected_defect_info)
        
        with tab2:
            st.header("Strip Map Analysis")

            if df_filtered.empty:
                st.info("먼저 사이드바에서 데이터를 필터링하세요.")
            else:
                available_strips = sorted(df_filtered['strip_id'].unique())
                if not available_strips:
                    st.warning("선택된 데이터에 분석할 Strip이 없습니다.")
                else:
                    selected_strip = st.selectbox("분석할 Strip ID를 선택하세요:", available_strips)

                    if selected_strip:
                        strip_data = df_filtered[df_filtered['strip_id'] == selected_strip]
                        
                        if 'n_unit_x' in strip_data.columns and 'n_unit_y' in strip_data.columns:
                            # --- 1. Strip Map 시각화 ---
                            st.subheader(f"Defect Count Map for Strip: `{selected_strip}`")
                            heatmap_pivot = strip_data.pivot_table(index='n_unit_y', columns='n_unit_x', values='has_defect', aggfunc='sum').fillna(0)
                            
                            fig = px.imshow(
                                heatmap_pivot,
                                labels=dict(x="Unit X", y="Unit Y", color="Defect Count"),
                                text_auto=True, color_continuous_scale='Reds', aspect="auto"
                            )
                            fig.update_layout(
                                title=f'Strip Map for {selected_strip}', yaxis_title="Unit Y",
                                height=600, xaxis=dict(side='top', title_text="Unit X")
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # --- [추가] 2. 선택된 Strip의 전체 불량 유형 요약 ---
                            st.divider()
                            st.subheader(f"Overall Defect Summary for Strip: `{selected_strip}`")
                            
                            defect_col_name = 'afvi_ai_defect'
                            strip_defects = strip_data[strip_data['has_defect'] == 1]

                            if strip_defects.empty:
                                st.success("해당 Strip에는 검출된 불량이 없습니다.")
                            else:
                                defect_summary = strip_defects[defect_col_name].value_counts().reset_index()
                                defect_summary.columns = ['Defect Type', 'Count']
                                st.dataframe(defect_summary, use_container_width=True, hide_index=True)
                            
                            # --- 3. 좌표별 드릴다운 기능 ---
                            st.divider()
                            st.subheader("Drill-Down by Unit Coordinates")
                            drill_cols = st.columns([1, 1, 2])
                            
                            max_x = int(strip_data['n_unit_x'].max()) if not strip_data['n_unit_x'].empty else 0
                            max_y = int(strip_data['n_unit_y'].max()) if not strip_data['n_unit_y'].empty else 0

                            x_coord = drill_cols[0].number_input("Unit X", min_value=0, max_value=max_x, step=1)
                            y_coord = drill_cols[1].number_input("Unit Y", min_value=0, max_value=max_y, step=1)

                            if drill_cols[2].button("View Defect Details", use_container_width=True):
                                unit_defects = strip_data[
                                    (strip_data['n_unit_x'] == x_coord) &
                                    (strip_data['n_unit_y'] == y_coord) &
                                    (strip_data['has_defect'] == 1)
                                ]
                                if unit_defects.empty:
                                    st.success(f"Unit ({x_coord}, {y_coord})에는 검출된 불량이 없습니다.")
                                else:
                                    st.write(f"Defects at Unit ({x_coord}, {y_coord}):")
                                    st.dataframe(
                                        unit_defects[['afvi_ai_defect', 'image_path']], 
                                        use_container_width=True, 
                                        hide_index=True
                                    )
                        else:
                            st.warning("Strip Map을 그리는 데 필요한 'n_unit_x' 또는 'n_unit_y' 컬럼을 찾을 수 없습니다.")
        with tab3:
            st.header("Image Viewer & Analysis")

            # [추가] 상세 정보 패널의 가독성 개선을 위한 CSS
            st.markdown("""
            <style>
                /* 비활성화된 text_input 스타일 재정의 */
                div[data-testid="stTextInput"] input[disabled] {
                    -webkit-text-fill-color: #FFFFFF; /* 값(value) 텍스트 색상을 흰색으로 */
                    color: #FFFFFF;
                    background-color: #1a1a1a; /* 배경색을 약간 더 어둡게 */
                }
                /* text_input의 라벨 색상 변경 */
                div[data-testid="stTextInput"] label {
                    color: #9A9A9A; /* 라벨 텍스트 색상을 밝은 회색으로 */
                }
            </style>
            """, unsafe_allow_html=True)

            if df_filtered.empty:
                st.info("먼저 사이드바에서 데이터를 필터링하세요.")
            else:
                image_paths = df_filtered.dropna(subset=['image_path'])['image_path'].unique()
                
                if len(image_paths) > 0:
                    selected_path = st.selectbox(
                        "분석할 이미지 경로(image_path)를 선택 또는 입력하세요:",
                        options=image_paths,
                        index=None,
                        placeholder="Choose an image path to analyze"
                    )
                else:
                    st.info("필터링된 데이터에 분석할 이미지가 없습니다.")
                    selected_path = None
                
                st.divider()

                if selected_path:
                    col1, col2 = st.columns([2, 3])

                    with col1:
                        st.subheader("Image Preview")
                        # (이미지 표시 로직은 기존과 동일)
                        st.markdown(
                            """
                            <div style="height: 400px; border: 2px dashed #555; display: flex; align-items: center; justify-content: center; border-radius: 0.5rem;">
                                <span style="color: #555;">Image Placeholder</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with col2:
                        st.subheader("Detailed Analysis")
                        
                        image_data = df_filtered[df_filtered['image_path'] == selected_path].iloc[0]
                        
                        # st.metric 대신 st.text_input(disabled=True)를 사용하여 UI 통일
                        with st.container(border=True):
                            st.markdown("##### 판정 정보 (Judgement)")
                            judgement_cols = st.columns(2)
                            judgement_cols[0].text_input("AI 판정", value=str(image_data.get('afvi_ai_defect', 'N/A')), disabled=True)
                            judgement_cols[1].text_input("작업자 판정", value=str(image_data.get('ivs_keyin1', 'N/A')), disabled=True)
                        
                        with st.container(border=True):
                            st.markdown("##### AI 신뢰도 점수 (AI Scores)")
                            score_cols = st.columns(3)
                            conf_score = image_data.get('afvi_clf_score', 0)
                            false_score = image_data.get('afvi_false_score', 0)
                            ng_score = image_data.get('afvi_ai_ng_score', 0)
                            score_cols[0].text_input("Confidence Score", value=f"{conf_score:.2%}" if pd.notna(conf_score) else "N/A", disabled=True)
                            score_cols[1].text_input("False Score", value=f"{false_score:.4f}" if pd.notna(false_score) else "N/A", disabled=True)
                            score_cols[2].text_input("NG Score", value=f"{ng_score:.4f}" if pd.notna(ng_score) else "N/A", disabled=True)

                        with st.container(border=True):
                            st.markdown("##### 물리적 특성 (Physical Properties)")
                            prop_cols = st.columns(3)
                            prop_cols[0].text_input("Defect Size (px)", value=str(image_data.get('defect_size', 'N/A')), disabled=True)
                            prop_cols[1].text_input("Defect Area", value=f"{image_data.get('afvi_ai_area', 0):.2f}", disabled=True)
                            prop_cols[2].text_input("Longest Axis", value=str(image_data.get('afvi_ai_longest', 'N/A')), disabled=True)

                        with st.container(border=True):
                            st.markdown("##### Gray Value (GV)")
                            gv_cols = st.columns(3)
                            gv_cols[0].text_input("AI GV", value=str(image_data.get('afvi_ai_gv', 'N/A')), disabled=True)
                            gv_cols[1].text_input("Otsu GV", value=str(image_data.get('afvi_otsu_gv', 'N/A')), disabled=True)
                            gv_cols[2].text_input("GV Gap", value=str(image_data.get('gv_gap', 'N/A')), disabled=True)
                        
                        if st.button("이 분석 결과 리포트하기", use_container_width=True, type="primary"):
                            report_data = {'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'image_path': selected_path, **image_data.to_dict()}
                            report_df = pd.DataFrame([report_data])
                            report_df.to_csv("analysis_report.csv", mode='a', header=not os.path.exists("analysis_report.csv"), index=False)
                            st.success("결과가 analysis_report.csv 파일에 저장되었습니다!")
        with tab4:
            st.header("Trend Analysis Dashboard")

            if df_filtered.empty:
                st.info("먼저 사이드바에서 데이터를 필터링하세요.")
            else:
                # 화면을 좌/우 두 개의 컬럼으로 분할
                col1, col2 = st.columns(2)

                # --- 1. 왼쪽 컬럼: 불량 유형 파레토 분석 ---
                with col1:
                    st.subheader("Defect Type Analysis (Pareto)")
                    
                    defect_col = 'afvi_ai_defect'
                    if defect_col in df_filtered.columns:
                        # 불량이 있는 데이터만 선택
                        defects_df = df_filtered[df_filtered['has_defect'] == 1]
                        
                        if defects_df.empty:
                            st.success("선택된 데이터에 불량이 없습니다.")
                        else:
                            # 불량 유형별로 개수 집계
                            defect_counts = defects_df[defect_col].value_counts().reset_index()
                            defect_counts.columns = [defect_col, 'count']
                            
                            # 파레토 차트 (막대 그래프) 생성
                            fig_pareto = px.bar(
                                defect_counts,
                                x='count',
                                y=defect_col,
                                orientation='h',
                                title='불량 유형별 발생 빈도 (가장 많은 순)',
                                labels={'count': '불량 수', defect_col: '불량 유형'}
                            )
                            # y축을 카운트 순으로 정렬하여 파레토 차트 효과
                            fig_pareto.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_pareto, use_container_width=True)
                    else:
                        st.warning(f"'{defect_col}' 컬럼을 찾을 수 없습니다.")

                # --- 2. 오른쪽 컬럼: 설비별 성능 비교 ---
                with col2:
                    st.subheader("Performance by Equipment")

                    equip_col = 'inspection_machine'
                    if equip_col in df_filtered.columns:
                        # 장비별로 전체 검사 수와 불량 수 집계
                        equip_summary = df_filtered.groupby(equip_col, observed=False).agg(
                            total_count=(equip_col, 'size'),
                            defect_count=('has_defect', 'sum')
                        ).reset_index()
                        
                        # 장비별 불량률 계산
                        equip_summary['defect_rate'] = (equip_summary['defect_count'] / equip_summary['total_count']) * 100
                        equip_summary = equip_summary.sort_values('defect_rate', ascending=True)

                        if equip_summary.empty:
                            st.info("분석할 장비 데이터가 없습니다.")
                        else:
                            # 장비별 불량률 비교 막대그래프 생성
                            fig_equip = px.bar(
                                equip_summary,
                                x='defect_rate',
                                y=equip_col,
                                orientation='h',
                                title='설비별 평균 불량률 비교',
                                labels={'defect_rate': '평균 불량률 (%)', equip_col: '장비'},
                                text=equip_summary['defect_rate'].apply(lambda x: f'{x:.2f}%') # 막대 위에 텍스트 표시
                            )
                            st.plotly_chart(fig_equip, use_container_width=True)
                    else:
                        st.warning(f"'{equip_col}' 컬럼을 찾을 수 없습니다.")

# --- 최종 정리 ---
gc.collect()