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
st.title("Inspection AI Inference result analysis system")

if 'filtered_data' not in st.session_state: st.session_state.filtered_data = None

with st.spinner('Loading and optimizing data...'):
    df_raw = load_data()

if df_raw.empty:
    st.warning("데이터를 불러오지 못했습니다. DB 연결을 확인해주세요.")
else:
    if 'display_item_code' not in df_raw.columns and 'image_path' in df_raw.columns:
        df_raw['display_item_code'] = df_raw['image_path'].apply(extract_item_code).astype('category')
    
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
            st.header("Strip Map")

            if df_filtered.empty:
                st.info("먼저 사이드바에서 데이터를 필터링하세요.")
            else:
                # Strip Map을 그릴 strip_id를 선택할 수 있도록 selectbox 추가
                available_strips = sorted(df_filtered['strip_id'].unique())
                selected_strip = st.selectbox("분석할 Strip ID를 선택하세요:", available_strips)

                if selected_strip:
                    # 선택된 strip_id에 해당하는 데이터만 추출
                    strip_data = df_filtered[df_filtered['strip_id'] == selected_strip]
                    
                    # [주의] 'n_unit_x', 'n_unit_y' 컬럼이 스트립 내 좌표라고 가정합니다.
                    if 'n_unit_x' in strip_data.columns and 'n_unit_y' in strip_data.columns:
                        
                        st.subheader(f"Defect Count Map for Strip: `{selected_strip}`")

                        # 각 Unit 좌표(n_unit_x, n_unit_y)별로 불량(has_defect) 개수를 집계
                        defect_map_data = strip_data.groupby(['n_unit_y', 'n_unit_x'])['has_defect'].sum().reset_index()
                        
                        # 히트맵을 그리기 위해 데이터를 피벗 테이블 형태로 변환
                        # 행(index)은 y좌표, 열(columns)은 x좌표, 값(values)은 불량 개수
                        heatmap_pivot = defect_map_data.pivot(index='n_unit_y', columns='n_unit_x', values='has_defect').fillna(0)

                        # Plotly를 사용하여 히트맵 생성
                        fig = px.imshow(
                            heatmap_pivot,
                            labels=dict(x="Unit X-Coordinate", y="Unit Y-Coordinate", color="Defect Count"),
                            x=heatmap_pivot.columns,
                            y=heatmap_pivot.index,
                            text_auto=True,  # 각 셀에 숫자 표시
                            color_continuous_scale='Reds' # 색상 스케일
                        )

                        fig.update_layout(
                            title=f'Strip Map for {selected_strip}',
                            xaxis_title="Unit X",
                            yaxis_title="Unit Y",
                            xaxis=dict(side='top') # x축을 위로 이동
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.warning("Strip Map을 그리는 데 필요한 'n_unit_x' 또는 'n_unit_y' 컬럼을 찾을 수 없습니다.")
        with tab3:
            st.header("Image Viewer & Analysis")

            if df_filtered.empty:
                st.info("먼저 사이드바에서 데이터를 필터링하세요.")
            else:
                image_paths = df_filtered.dropna(subset=['image_path'])['image_path'].unique()
                
                if len(image_paths) > 0:
                    selected_path = st.selectbox(
                        "분석할 이미지 경로(image_path)를 선택하세요:",
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

                        # [추가] AI 판정과 작업자 판정을 각각 변수에 저장
                        # [주의] 작업자 판정 컬럼명은 'ivs_keyin1'로 가정했습니다. 실제 컬럼명으로 수정이 필요할 수 있습니다.
                        ai_judgment = image_data.get('afvi_ai_defect', 'N/A')
                        human_judgment = image_data.get('ivs_keyin1', 'N/A')
                        
                        st.text_input("AI Judgment (afvi_ai_defect)", value=ai_judgment, disabled=True)
                        # [추가] 작업자 판정 정보 표시
                        st.text_input("Human Judgment (ivs_keyin1)", value=human_judgment, disabled=True)

                        # (이하 다른 정보들은 기존과 동일)
                        confidence_score = image_data.get('afvi_clf_score', 0)
                        st.text_input("Confidence Score", value=f"{confidence_score:.2%}" if pd.notna(confidence_score) else "N/A", disabled=True)
                        st.text_input("Defect Size (defect_size)", value=image_data.get('defect_size', 'N/A'), disabled=True)
                        st.text_area("Full Image Path", value=selected_path, height=100, disabled=True)
                        
                        st.divider()

                        # [추가] AI와 작업자 판정 비교 및 버튼 생성
                        st.subheader("Judgment Comparison")

                        # NaN 값을 비교하기 위해 문자열로 변환하여 비교
                        is_match = str(ai_judgment).strip() == str(human_judgment).strip()

                        if is_match:
                            st.success("✅ AI and Human Judgments Match")
                        else:
                            st.error("⚠️ AI and Human Judgments DO NOT Match")
                            # 불일치할 경우에만 리포트 버튼 표시
                            if st.button("불일치 내역 리포트", use_container_width=True, type="primary"):
                                st.toast(f"불일치 리포트가 기록되었습니다: {os.path.basename(selected_path)}")
                                # 실제 앱에서는 이 피드백을 로그나 DB에 저장하는 로직을 추가합니다.
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