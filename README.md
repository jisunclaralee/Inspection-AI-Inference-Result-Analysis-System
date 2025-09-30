# Inspection AI Inference Result Analysis System

📊 **PCB 검사 AI 결과 분석 대시보드**

## 🎯 개요

이 프로그램은 PCB(인쇄회로기판) 검사 AI의 결과를 분석하고 시각화하는 대시보드입니다. PostgreSQL 데이터베이스에서 검사 데이터를 가져와서 실시간으로 분석 결과를 제공합니다.

## ✨ 주요 기능

### 📋 **LOT ANALYSIS**
- **드릴다운 분석**: Test ID → Bundle → Strip → Defect → Image Path 순으로 상세 분석
- **실시간 필터링**: 날짜, 장비, 고객사, Lot, Bundle 등 다양한 조건으로 필터링
- **이미지 팝업**: 불량 이미지를 클릭하면 팝업으로 확인 가능

### 🗺️ **STRIP MAP**
- **히트맵 시각화**: Strip 내 불량 분포를 히트맵으로 표시
- **좌표별 드릴다운**: 특정 좌표의 불량 상세 정보 확인
- **불량 유형 요약**: Strip 전체의 불량 유형별 통계

### 🖼️ **IMAGE VIEWER**
- **상세 분석**: 개별 이미지의 AI 판정, 신뢰도 점수, 물리적 특성 등 확인
- **비교 분석**: AI 판정 vs 작업자 판정 비교
- **리포트 생성**: 분석 결과를 CSV 파일로 저장

### 📈 **TREND ANALYSIS**
- **파레토 분석**: 불량 유형별 발생 빈도 분석
- **장비별 성능**: 검사 장비별 불량률 비교 분석

## 🚀 실행 방법

### 방법 1: EXE 파일 실행 (권장)
1. `InspectionAI_Dashboard.exe` 파일을 더블클릭
2. 콘솔 창이 열리면서 앱이 시작됩니다
3. 자동으로 브라우저에서 대시보드가 열립니다
4. 앱을 종료하려면 콘솔 창을 닫으세요

### 방법 2: Python 스크립트 실행
```bash
# 가상환경 활성화
& "C:\Users\user\Desktop\새 폴더\venv\Scripts\Activate.ps1"

# 앱 실행
python -m streamlit run "Inspection AI4.py"
```

## 🔧 시스템 요구사항

### **필수 요구사항**
- **Windows 10/11** (64비트)
- **PostgreSQL 데이터베이스** 실행 중이어야 함
- **최소 8GB RAM** (권장: 16GB 이상)

### **데이터베이스 설정**
데이터베이스 연결 정보는 `.streamlit/secrets.toml` 파일에 설정되어 있습니다:

```toml
[database]
host = "localhost"
port = 5432
dbname = "postgres"
user = "postgres"
password = "0821"
```

### **필요한 데이터 테이블**
- 테이블명: `public.inspection_data`
- 주요 컬럼:
  - `test_id`, `lot_no`, `bundle_no`, `strip_id`
  - `ai_date_time`, `inspection_machine`, `rms_customer`
  - `afvi_ai_defect`, `has_defect`, `image_path`
  - `n_unit_x`, `n_unit_y` (Strip Map용)
  - AI 분석 결과 컬럼들

## 📁 파일 구조

```
📦 Inspection-AI-Analysis-System/
├── 🗂️ .streamlit/
│   ├── config.toml          # Streamlit 설정
│   └── secrets.toml         # 데이터베이스 연결 정보
├── 📄 Inspection AI4.py     # 메인 대시보드 코드
├── 📄 app_launcher.py       # EXE 런처 스크립트
├── 📄 app_launcher.spec     # PyInstaller 설정
├── 📄 build_exe.bat         # EXE 빌드 스크립트
├── 📂 dist/
│   └── 🚀 InspectionAI_Dashboard.exe  # 실행 파일
└── 📄 README.md             # 사용 설명서
```

## 🛠️ 개발자 정보

### **빌드 방법**
```bash
# PyInstaller로 EXE 생성
pyinstaller --clean app_launcher.spec

# 또는 배치 파일 실행
build_exe.bat
```

### **의존성 패키지**
- streamlit
- pandas
- numpy
- plotly
- sqlalchemy
- psycopg2-binary
- streamlit-plotly-events

## 🔍 사용 팁

1. **성능 최적화**: 대량 데이터 분석 시 필터를 사용하여 데이터 범위를 줄이세요
2. **이미지 로딩**: 이미지 경로는 URL 또는 절대/상대 경로를 지원합니다
3. **메모리 관리**: 대시보드는 자동으로 메모리를 최적화합니다
4. **데이터 새로고침**: 브라우저에서 F5를 눌러 데이터를 새로고침할 수 있습니다

## 📞 문의사항

기술적인 문제나 개선사항이 있으시면 GitHub Issues를 통해 문의해주세요.

**Repository**: https://github.com/jisunclaralee/Inspection-AI-Inference-Result-Analysis-System

---

📅 **Last Updated**: 2025-09-30  
🏷️ **Version**: 1.0.0  
👤 **Developer**: jisunclaralee