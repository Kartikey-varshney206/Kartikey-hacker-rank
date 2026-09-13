@echo off
setlocal

cd /d "%~dp0"
set "PYTHON_RUNTIME=C:\Users\karti\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if not exist "%PYTHON_RUNTIME%" (
  echo Bundled Python runtime was not found:
  echo %PYTHON_RUNTIME%
  echo Update PYTHON_RUNTIME in this file, then run it again.
  pause
  exit /b 1
)

echo Running the sample-only Groq pipeline...
"%PYTHON_RUNTIME%" code\main.py --input sample --llm
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo The pipeline stopped with exit code %EXIT_CODE%.
) else (
  echo.
  echo Finished. Review sample_output.csv and evaluation\sample_report.md.
)

pause
exit /b %EXIT_CODE%
