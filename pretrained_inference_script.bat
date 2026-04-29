@echo off
setlocal

REM BASE MODEL
set BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.2

REM PARAMETERS
set maxlen=5120
set mem=128
set r=512
set mean_compression_rate=4

REM ICAE model path passed as first argument
REM Example:
REM run_inference.bat mistral_7b_pretrained_icae.safetensors

set ICAE_MODEL_PATH=model/mistral_7b_pretrained_icae.safetensors

if "%ICAE_MODEL_PATH%"=="" (
    echo Error: Please provide ICAE model path as first argument.
    echo Example:
    echo run_inference.bat mistral_7b_pretrained_icae.safetensors
    pause
    exit /b 1
)

python pretrained_inference.py ^
    --mean_compression_rate %mean_compression_rate% ^
    --model_max_length %maxlen% ^
    --fixed_mem_size %mem% ^
    --lora_r %r% ^
    --output_dir %ICAE_MODEL_PATH% ^
    --model_name_or_path %BASE_MODEL% ^
    --bf16 ^
    --train False

endlocal