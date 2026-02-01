@echo off
setlocal EnableDelayedExpansion

:: ==========================================
:: Windows Batch Script for Fed3DGS Training
:: Functionally equivalent to client_training.sh
:: ==========================================

set COLMAP_RESULTS_DIR=%3
set DATASET_ROOT=%4
set OUTPUT_DIR=%5

for /L %%i in (%1, 1, %2) do (
    :: Format number with leading zeros (0 -> 00000 for 5-digit client ID)
    set "NUM=00000%%i"
    set "SEQ_ID=!NUM:~-5!"
    
    python gaussian-splatting\train.py -s "!COLMAP_RESULTS_DIR!\!SEQ_ID!" -i "!DATASET_ROOT!\train\rgbs" -w -m "!OUTPUT_DIR!\!SEQ_ID!"
)

endlocal