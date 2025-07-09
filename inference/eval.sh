MODEL_PATH=your_model
SAMPLE_NUM=1
POST_BEAM_SIZE=2
SAMPLE_TEMP=1
OUTPUT_PATH=your_output_path
INPUT_DATA_PATH=your_data
GPU_IDS=0,1,2,3,4,5,6,7,8

python eval.py \
    --model_name_or_path ${MODEL_PATH} \
    --max_new_tokens 2048 \
    --input_file ${INPUT_DATA_PATH} \
    --output_file ${OUTPUT_PATH}/sample_output.jsonl \
    --sample_num ${SAMPLE_NUM} \
    --temperature ${SAMPLE_TEMP} \
    --gpu_ids ${GPU_IDS}

python prepare_beam_search.py \
    --tokenizer_path ${MODEL_PATH} \
    --input_file ${OUTPUT_PATH}/sample.jsonl \
    --output_file ${OUTPUT_PATH}/beam_search_input.jsonl

python eval.py \
    --model_name_or_path ${MODEL_PATH} \
    --max_new_tokens 256 \
    --input_file ${OUTPUT_PATH}/beam_search_input.jsonl \
    --output_file ${OUTPUT_PATH}/beam_search_output.jsonl \
    --sample_num ${POST_BEAM_SIZE} \
    --temperature 0 \
    --gpu_ids ${GPU_IDS} \
    --use_beam_search \
    --post_beam 

python metric.py --pred ${OUTPUT_PATH}/beam_search_output.jsonl