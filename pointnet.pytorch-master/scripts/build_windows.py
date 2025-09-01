#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows平台C++编译脚本
用于编译PointNet可视化所需的C++扩展模块

支持多种编译器：
1. Visual Studio (cl.exe)
2. MinGW-w64 (g++.exe)
3. MSVC (通过conda环境)

使用方法：
python scripts/build_windows.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_compiler(compiler_name):
    """检查编译器是否可用"""
    try:
        result = subprocess.run([compiler_name, '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def find_visual_studio():
    """查找Visual Studio编译器"""
    # 常见的Visual Studio安装路径
    vs_paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
    ]
    
    for vs_path in vs_paths:
        if os.path.exists(vs_path):
            # 查找最新版本的MSVC
            versions = [d for d in os.listdir(vs_path) if d.startswith('14.')]
            if versions:
                latest_version = sorted(versions)[-1]
                cl_path = os.path.join(vs_path, latest_version, 'bin', 'Hostx64', 'x64', 'cl.exe')
                if os.path.exists(cl_path):
                    return cl_path
    return None

def compile_with_msvc(cpp_file, output_file):
    """使用Visual Studio编译器编译"""
    cl_path = find_visual_studio()
    if not cl_path:
        print("❌ 未找到Visual Studio编译器")
        return False
    
    # 设置环境变量
    vs_path = os.path.dirname(os.path.dirname(os.path.dirname(cl_path)))
    vcvars_path = os.path.join(vs_path, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat')
    
    if not os.path.exists(vcvars_path):
        print(f"❌ 未找到vcvars64.bat: {vcvars_path}")
        return False
    
    # 构建编译命令
    cmd = f'"{vcvars_path}" && cl /LD /std:c++11 /O2 "{cpp_file}" /Fe:"{output_file}"'
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 使用Visual Studio编译成功")
            return True
        else:
            print(f"❌ Visual Studio编译失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Visual Studio编译异常: {e}")
        return False

def compile_with_mingw(cpp_file, output_file):
    """使用MinGW编译器编译"""
    try:
        cmd = ['g++', '-std=c++11', '-shared', '-fPIC', '-O2', cpp_file, '-o', output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 使用MinGW编译成功")
            return True
        else:
            print(f"❌ MinGW编译失败: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ 未找到MinGW编译器(g++)")
        return False

def main():
    print("🔧 PointNet C++扩展编译工具 (Windows)")
    print("=" * 50)
    
    # 获取路径
    script_dir = Path(__file__).parent
    utils_dir = script_dir.parent / "utils"
    cpp_file = utils_dir / "render_balls_so.cpp"
    output_file = utils_dir / "render_balls_so.dll"
    
    print(f"📁 脚本目录: {script_dir}")
    print(f"📁 工具目录: {utils_dir}")
    print(f"📄 源文件: {cpp_file}")
    print(f"📄 输出文件: {output_file}")
    print()
    
    # 检查源文件是否存在
    if not cpp_file.exists():
        print(f"❌ 源文件不存在: {cpp_file}")
        return 1
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("🔍 检测可用的编译器...")
    
    # 尝试不同的编译方法
    success = False
    
    # 方法1: 尝试MinGW
    print("\n1️⃣ 尝试使用MinGW编译器...")
    if check_compiler('g++'):
        success = compile_with_mingw(str(cpp_file), str(output_file))
    
    # 方法2: 尝试Visual Studio
    if not success:
        print("\n2️⃣ 尝试使用Visual Studio编译器...")
        success = compile_with_msvc(str(cpp_file), str(output_file))
    
    # 方法3: 尝试conda环境中的编译器
    if not success:
        print("\n3️⃣ 尝试使用conda环境中的编译器...")
        try:
            # 检查是否在conda环境中
            conda_prefix = os.environ.get('CONDA_PREFIX')
            if conda_prefix:
                conda_cl = os.path.join(conda_prefix, 'Library', 'bin', 'cl.exe')
                if os.path.exists(conda_cl):
                    cmd = [conda_cl, '/LD', '/std:c++11', '/O2', str(cpp_file), f'/Fe:{output_file}']
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print("✅ 使用conda环境编译器编译成功")
                        success = True
        except Exception as e:
            print(f"❌ conda环境编译失败: {e}")
    
    if success:
        print(f"\n🎉 编译成功！输出文件: {output_file}")
        print("\n💡 提示：")
        print("   - 如果遇到导入错误，请确保DLL文件在Python路径中")
        print("   - 某些Python包可能需要重新安装以识别新的DLL")
        return 0
    else:
        print("\n❌ 所有编译方法都失败了")
        print("\n🔧 解决方案：")
        print("1. 安装Visual Studio Build Tools:")
        print("   https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022")
        print("2. 安装MinGW-w64:")
        print("   https://www.mingw-w64.org/downloads/")
        print("3. 使用MSYS2:")
        print("   https://www.msys2.org/")
        print("4. 或者跳过可视化功能，直接运行训练脚本")
        return 1

if __name__ == "__main__":
    sys.exit(main())
