@echo off
echo ========================================
echo Inspection AI Dashboard EXE Builder
echo ========================================

:: 가상환경 활성화
call "C:\Users\user\Desktop\새 폴더\venv\Scripts\activate.bat"

:: 현재 디렉토리로 이동
cd /d "C:\Users\user\Desktop\새 폴더\venv\pcb_dashboard\src"

echo.
echo 📦 PyInstaller로 EXE 파일 생성 중...
echo.

:: PyInstaller로 EXE 생성
pyinstaller --clean app_launcher.spec

echo.
if exist "dist\InspectionAI_Dashboard.exe" (
    echo ✅ EXE 파일 생성 완료!
    echo 📁 위치: %cd%\dist\InspectionAI_Dashboard.exe
    echo.
    echo 💡 사용법:
    echo    1. InspectionAI_Dashboard.exe 를 실행하세요
    echo    2. 브라우저가 자동으로 열립니다
    echo    3. PostgreSQL 데이터베이스가 실행 중인지 확인하세요
    echo.
    
    :: 파일 탐색기에서 dist 폴더 열기
    explorer "dist"
) else (
    echo ❌ EXE 파일 생성 실패
    echo 오류 로그를 확인하세요.
)

echo.
pause