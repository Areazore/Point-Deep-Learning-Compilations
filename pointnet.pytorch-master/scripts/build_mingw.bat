@echo off
REM Windows批处理脚本：使用MinGW编译C++可视化代码
REM 使用方法：在scripts目录下运行 build_mingw.bat

set SCRIPT_PATH=%~dp0
set UTILS_PATH=%SCRIPT_PATH%..\utils

echo 正在使用MinGW编译C++可视化代码...
echo 脚本路径: %SCRIPT_PATH%
echo 工具路径: %UTILS_PATH%

REM 检查是否安装了MinGW编译器
where g++ >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误：未找到MinGW编译器(g++.exe)
    echo 请确保已安装MinGW-w64或MSYS2
    echo 下载地址：https://www.mingw-w64.org/downloads/
    echo 或者使用MSYS2：https://www.msys2.org/
    pause
    exit /b 1
)

REM 编译为DLL文件
g++ -std=c++11 -shared -fPIC -O2 "%UTILS_PATH%\render_balls_so.cpp" -o "%UTILS_PATH%\render_balls_so.dll"

if %errorlevel% equ 0 (
    echo 编译成功！生成文件：%UTILS_PATH%\render_balls_so.dll
) else (
    echo 编译失败！
    pause
    exit /b 1
)

pause
