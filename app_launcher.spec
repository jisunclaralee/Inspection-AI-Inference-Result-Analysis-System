# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 현재 디렉토리
current_dir = os.path.dirname(os.path.abspath('app_launcher.py'))

# Streamlit 관련 데이터 파일들 수집
streamlit_datas = collect_data_files('streamlit')

# 추가 데이터 파일들
added_files = [
    ('Inspection AI4.py', '.'),
    ('.streamlit', '.streamlit'),
]

# 숨겨진 imports 수집
hidden_imports = [
    'streamlit',
    'pandas',
    'numpy',
    'plotly',
    'plotly.express',
    'plotly.graph_objects',
    'sqlalchemy',
    'psycopg2',
    'streamlit_plotly_events',
    'PIL',
    'PIL.Image',
    'altair',
    'pyarrow',
    'pydeck',
    'tornado',
    'packaging',
    'packaging.version',
    'packaging.specifiers',
    'packaging.requirements',
]

# Streamlit 서브모듈들 수집
streamlit_modules = collect_submodules('streamlit')
hidden_imports.extend(streamlit_modules)

block_cipher = None

a = Analysis(
    ['app_launcher.py'],
    pathex=[current_dir],
    binaries=[],
    datas=streamlit_datas + added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='InspectionAI_Dashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 여기에 .ico 파일 경로를 추가할 수 있습니다
)