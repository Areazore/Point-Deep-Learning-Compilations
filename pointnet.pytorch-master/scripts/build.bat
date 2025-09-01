@echo off
REM Windows批处理脚本：编译C++可视化代码
REM 使用方法：在scripts目录下运行 build.bat

set SCRIPT_PATH=%~dp0
set UTILS_PATH=%SCRIPT_PATH%..\utils

echo 正在编译C++可视化代码...
echo 脚本路径: %SCRIPT_PATH%
echo 工具路径: %UTILS_PATH%

REM 检查是否安装了Visual Studio编译器
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误：未找到Visual Studio编译器(cl.exe)
    echo 请确保已安装Visual Studio或Visual Studio Build Tools
    echo 或者运行以下命令设置环境变量：
    echo "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    pause
    exit /b 1
)

REM 编译为DLL文件
cl /LD /std:c++11 /O2 "%UTILS_PATH%\render_balls_so.cpp" /Fe:"%UTILS_PATH%\render_balls_so.dll"

if %errorlevel% equ 0 (
    echo 编译成功！生成文件：%UTILS_PATH%\render_balls_so.dll
) else (
    echo 编译失败！
    pause
    exit /b 1
)

pause
