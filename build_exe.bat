@echo off
echo ============================================
echo   Building Research Paper Explainer EXE
echo ============================================
echo.

REM Install PyInstaller if needed
pip install pyinstaller

echo.
echo Building EXE...
pyinstaller launcher.spec --clean

echo.
echo ============================================
echo   Build complete!
echo   Output: dist\ResearchPaperExplainer\
echo   Run:    dist\ResearchPaperExplainer\ResearchPaperExplainer.exe
echo ============================================
pause
