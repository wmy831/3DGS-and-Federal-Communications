@echo off
setlocal EnableDelayedExpansion

:: ==========================================
:: Windows Batch Script for Fed3DGS Training
:: (English only to avoid encoding errors)
:: ==========================================

if "%~6"=="" (
    echo Usage: scripts\client_training.bat start_idx end_idx colmap_dir dataset_root image_list_dir output_dir
    exit /b 1
)

set START_IDX=%1
set END_IDX=%2
set COLMAP_RESULTS_DIR=%3
set DATASET_ROOT=%4
set IMAGE_LIST_DIR=%5
set OUTPUT_DIR=%6

:: Loop from START_IDX to END_IDX
for /L %%i in (%START_IDX%, 1, %END_IDX%) do (
    
    :: Format number with leading zeros (0 -> 00000)
    set "NUM=00000%%i"
    set "SEQ_ID=!NUM:~-5!"
    
    echo.
    echo ========================================
    echo Processing Sequence: !SEQ_ID!
    echo ========================================

    set "CUR_COLMAP_DIR=!COLMAP_RESULTS_DIR!\!SEQ_ID!"
    set "TRAIN_DIR=!DATASET_ROOT!\train"
    set "IMG_LIST=!IMAGE_LIST_DIR!\!SEQ_ID!.txt"
    set "CUR_OUTPUT_DIR=!OUTPUT_DIR!\!SEQ_ID!"
    set "RGBS_DIR=!TRAIN_DIR!\rgbs"

    :: Step 1: Try to run triangulation
    :: If tools\triangulate_colmap.bat exists, run it.
    if exist "tools\triangulate_colmap.bat" (
        echo Running triangulation...
        call tools\triangulate_colmap.bat "!CUR_COLMAP_DIR!" "!TRAIN_DIR!" "!IMG_LIST!"
    ) else (
        echo [WARNING] tools\triangulate_colmap.bat not found. Skipping triangulation.
        echo Ensure your COLMAP data is already prepared.
    )

    :: Step 2: Run Gaussian Splatting Training
    echo Running training...
    python gaussian-splatting\train.py -s "!CUR_COLMAP_DIR!" -i "!RGBS_DIR!" -w -m "!CUR_OUTPUT_DIR!" 
    ::--iterations 3000 -r 2
    :: --iterations 3000 : 将迭代次数限制为 3000 (默认通常是 30000)
    :: -r 2              : 对图像进行 2 倍下采样训练 (分辨率变小，速度变快)
    if errorlevel 1 (
        echo [ERROR] Training failed for sequence !SEQ_ID!
    )
)

echo All done.
endlocal