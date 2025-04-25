@echo off
REM Activate the virtual environment located in the env folder next to this script
call env\Scripts\activate
if errorlevel 1 goto error

REM Run the ColorRemapper script without opening a console window
pythonw ColorRemapper.py
if errorlevel 1 goto error

goto end

:error
echo An error occurred while running ColorRemapper.py.
pause

:end
