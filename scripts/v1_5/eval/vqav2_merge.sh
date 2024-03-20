gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl
# /liymai24/sjtu/bokai/LLaVA/playground/data/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/llava-v1.5-13b/merge.jsonl
#                          ./playground/data/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/llava-v1.5-13b/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Processing file: ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl"
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done