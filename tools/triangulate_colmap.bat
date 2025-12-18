@echo off
setlocal EnableDelayedExpansion

:: ==========================================
:: Triangulate COLMAP Script (Windows Port)
:: Based on Fed3DGS/tools/triangulate_colmap.sh
:: ==========================================

:: 参数定义
:: %1 = 输出目录 (Output Directory)
:: %2 = 数据集根目录/train (Dataset Root)
:: %3 = 图像列表文件 (Index File)

set "OUT_DIR=%~1"
set "DATA_ROOT=%~2"
set "INDEX_FILE=%~3"

:: 1. 创建数据库 (Create Database)
echo create database
python tools\create_db.py -out "%OUT_DIR%" -r "%DATA_ROOT%" --index-file "%INDEX_FILE%"
if errorlevel 1 goto :error

:: 2. 特征提取 (Feature Extractor)
colmap feature_extractor --database_path "%OUT_DIR%\database.db" --image_path "%DATA_ROOT%\rgbs" --image_list_path "%INDEX_FILE%" --ImageReader.camera_model PINHOLE
if errorlevel 1 goto :error

:: 3. 特征匹配 (Exhaustive Matcher)
colmap exhaustive_matcher --database_path "%OUT_DIR%\database.db"
if errorlevel 1 goto :error

:: 4. 创建 sparse/0 目录
if not exist "%OUT_DIR%\sparse\0" mkdir "%OUT_DIR%\sparse\0"

:: 5. 点三角化 (Point Triangulator) - 这是关键步骤！
colmap point_triangulator --database_path "%OUT_DIR%\database.db" --image_path "%DATA_ROOT%\rgbs" --input_path "%OUT_DIR%\sparse" --output_path "%OUT_DIR%\sparse\0" --Mapper.tri_ignore_two_view_tracks=0
if errorlevel 1 goto :error

:: 6. 清理临时文件 (Cleanup)
del /Q "%OUT_DIR%\sparse\*.txt" 2>nul
del /Q "%OUT_DIR%\database.db" 2>nul

echo [SUCCESS] Triangulation completed successfully.
exit /b 0

:error
echo [ERROR] An error occurred during COLMAP processing.
exit /b 1
