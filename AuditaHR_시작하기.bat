@echo off
chcp 65001 > nul
title AuditaHR - AI 채용 공정성 감사 및 교정 미들웨어 실행기
cls
echo ================================================================
echo      ⚖️ AuditaHR Enterprise v2.4 (원클릭 브라우저 런처)
echo ================================================================
echo.
echo [1/3] 파이썬 및 필수 패키지 환경을 확인하는 중입니다...
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] 파이썬(Python)이 설치되어 있지 않거나 환경 변수에 등록되지 않았습니다.
    echo 파이썬 3.10 이상을 설치한 후 다시 실행해 주세요.
    pause
    exit /b
)

echo [2/3] AuditaHR 웹 애플리케이션을 구동합니다...
echo       (잠시 후 기본 웹 브라우저에서 자동으로 링크가 열립니다.)
echo.
echo [3/3] 브라우저 접속 링크: http://localhost:8501
echo ================================================================
echo * 앱을 종료하려면 이 창을 닫거나 Ctrl + C 를 누르세요.
echo ================================================================
echo.

:: 기본 브라우저에서 링크 자동 오픈 및 streamlit 실행
start http://localhost:8501
streamlit run app.py --server.headless=true --server.port=8501
