model_array=("meta-llama/Llama-2-7b-hf" "microsoft/phi-2")
lora_array=("sft/Llama-2-7b-hf/alpaca" "sft/gemma-2b/alpaca")
# Need to prepross the data fisrt and then replace them with your own huggingface data path
data_array=("xxx/hhgolden" "xxx/oasst1") 
defense_array=("none" "cdpo" "rdpo" "ours" "ourscdpo" "oursrdpo")

train(){
    local model=1
    local data=1
    local defense_id=$1
    local epsilon=$2
    local seed=42
    
    local model_path=${model_array[model]}
    local lora_path=${lora_array[model]}
    local data_path=${data_array[data]}
    local defense=${defense_array[defense_id]}
    
    python src/ppo.py \
        --model_path  "${model_path}" \
        --data_path "${data_path}" \
        --save_dir "ppo" \
        --lora_path "${lora_path}" \
        --epsilon "${epsilon}" \
        --defense "${defense}" \
        --seed "${seed}"
}

(
    export CUDA_VISIBLE_DEVICES=0
    train 0 0.1
    train 1 0.1 
    train 2 0.1 
    train 3 0.1 
    train 4 0.1 
    train 5 0.1 
)&
(
    export CUDA_VISIBLE_DEVICES=1
    train 0 0.2
    train 1 0.2 
    train 2 0.2 
    train 3 0.2 
    train 4 0.2 
    train 5 0.2 
)&
(
    export CUDA_VISIBLE_DEVICES=2
    train 0 0.3
    train 1 0.3 
    train 2 0.3 
    train 3 0.3 
    train 4 0.3 
    train 5 0.3 
)&
(
    export CUDA_VISIBLE_DEVICES=3
    train 0 0.4
    train 1 0.4 
    train 2 0.4 
    train 3 0.4 
    train 4 0.4 
    train 5 0.4 
)&
wait