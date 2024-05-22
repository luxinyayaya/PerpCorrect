model_array=("meta-llama/Llama-2-7b-hf" "microsoft/phi-2")
data_array=("tatsu-lab/alpaca")

train(){
    local model=$1
    local data=$2

    local model_path=${model_array[model]}
    local data_path=${data_array[data]}
    
    python src/dpo.py \
    --model_path "${model_path}" \
    --data_path "${data_path}" \
    --save_dir "sft"
}

(
    export CUDA_VISIBLE_DEVICES=0
    train 0 0 
)&

wait