@echo off
setlocal EnableDelayedExpansion

:: ==========================================
:: Windows Batch Script for Fed3DGS Training
:: 使用已分配的点云（跳过三角化步骤）
:: ==========================================

if "%~6"=="" (
    echo Usage: scripts\client_training_with_pointcloud.bat start_idx end_idx colmap_dir dataset_root image_list_dir output_dir
    echo.
    echo Note: This script skips triangulation and uses pre-distributed point clouds.
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

    :: Check if COLMAP directory exists
    if not exist "!CUR_COLMAP_DIR!\sparse\0\points3D.bin" (
        if not exist "!CUR_COLMAP_DIR!\sparse\0\points3D.txt" (
            echo [ERROR] Point cloud not found for sequence !SEQ_ID!
            echo Expected: !CUR_COLMAP_DIR!\sparse\0\points3D.bin or .txt
            echo Skipping...
            goto :next_client
        )
    )

    echo Using pre-distributed point cloud from: !CUR_COLMAP_DIR!
    echo Skipping triangulation step.

    :: Step 2: Run Gaussian Splatting Training
    echo Running training...
    python gaussian-splatting\train.py -s "!CUR_COLMAP_DIR!" -i "!RGBS_DIR!" -w -m "!CUR_OUTPUT_DIR!" 
    if errorlevel 1 (
        echo [ERROR] Training failed for sequence !SEQ_ID!
    )

    :next_client
)

echo All done.
endlocal

