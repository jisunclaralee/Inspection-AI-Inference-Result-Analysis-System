#!/usr/bin/env python3
"""
Inspection AI Inference Result Analysis System - EXE Launcher
독립 실행 가능한 앱 런처
"""

import sys
import os
import subprocess
import webbrowser
import time
import socket
from threading import Thread
import tempfile
import shutil

def find_free_port():
    """사용 가능한 포트를 찾습니다."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def check_port_in_use(port):
    """포트가 사용 중인지 확인합니다."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def copy_streamlit_config():
    """Streamlit 설정 파일을 임시 위치로 복사합니다."""
    # 현재 실행 파일의 디렉토리
    if getattr(sys, 'frozen', False):
        # PyInstaller로 패키징된 경우
        app_dir = sys._MEIPASS
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # .streamlit 폴더 경로
    streamlit_dir = os.path.join(app_dir, '.streamlit')
    
    # 사용자 홈 디렉토리의 .streamlit 폴더로 복사
    home_streamlit_dir = os.path.expanduser('~/.streamlit')
    
    if os.path.exists(streamlit_dir):
        if not os.path.exists(home_streamlit_dir):
            os.makedirs(home_streamlit_dir)
        
        for file in ['config.toml', 'secrets.toml']:
            src = os.path.join(streamlit_dir, file)
            dst = os.path.join(home_streamlit_dir, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"복사됨: {file}")

def run_streamlit_app():
    """Streamlit 앱을 실행합니다."""
    
    # 설정 파일 복사
    copy_streamlit_config()
    
    # 사용 가능한 포트 찾기
    port = find_free_port()
    
    # 현재 스크립트의 디렉토리 찾기
    if getattr(sys, 'frozen', False):
        # PyInstaller로 패키징된 경우
        app_dir = sys._MEIPASS
        app_file = os.path.join(app_dir, 'Inspection AI4.py')
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        app_file = os.path.join(app_dir, 'Inspection AI4.py')
    
    print(f"🚀 Inspection AI Analysis System 시작 중...")
    print(f"📁 앱 디렉토리: {app_dir}")
    print(f"📄 앱 파일: {app_file}")
    print(f"🌐 포트: {port}")
    
    try:
        # Streamlit 명령 구성
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            app_file,
            '--server.port', str(port),
            '--server.address', 'localhost',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false',
            '--theme.base', 'dark'
        ]
        
        print(f"💻 실행 명령: {' '.join(cmd)}")
        
        # Streamlit 프로세스 시작
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=app_dir
        )
        
        # 앱이 시작될 때까지 대기
        print("⏳ 앱 시작 대기 중...")
        
        max_wait = 30  # 최대 30초 대기
        waited = 0
        
        while waited < max_wait:
            if check_port_in_use(port):
                break
            time.sleep(1)
            waited += 1
        
        if waited >= max_wait:
            print("❌ 앱 시작 시간 초과")
            return
        
        # 브라우저에서 앱 열기
        url = f"http://localhost:{port}"
        print(f"🌍 브라우저에서 앱 열기: {url}")
        webbrowser.open(url)
        
        print("✅ 앱이 성공적으로 시작되었습니다!")
        print("📱 브라우저에서 앱을 확인하세요.")
        print("🔄 앱을 종료하려면 이 창을 닫으세요.")
        
        # 프로세스 출력 모니터링
        for line in process.stdout:
            if line.strip():
                print(f"📊 {line.strip()}")
        
        # 프로세스 종료 대기
        process.wait()
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        input("Enter 키를 눌러 종료하세요...")

def main():
    """메인 함수"""
    print("=" * 60)
    print("🔬 Inspection AI Inference Result Analysis System")
    print("=" * 60)
    
    try:
        run_streamlit_app()
    except KeyboardInterrupt:
        print("\n⏹️  사용자가 앱을 종료했습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {str(e)}")
        input("Enter 키를 눌러 종료하세요...")

if __name__ == "__main__":
    main()