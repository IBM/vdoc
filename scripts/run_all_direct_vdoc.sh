HOME=/Users/yosimass/PycharmProjects/data/scrolls
ORDER=rank
SEMANTICS=sliding-window
PASSAGE_SIZE=512
MAX_NEW_TOKENS=0
SPLIT=validation
#
DATASETS=("qasper" "narrativeqa")
MODELS=("prefix" "google/flan-t5-large" "sentence-transformers/all-MiniLM-L12-v2" "elser")
WINDOWS=("2400" "4800" "7200")
for dataset in "${DATASETS[@]}"
do
  echo ${dataset}
  for model in "${MODELS[@]}"
  do
    echo ${model}
    for window in "${WINDOWS[@]}"
    do
      echo ${window}
      PYTHONPATH=. python -u vdoc_eval.py \
      --dataset scrolls \
      --order ${ORDER} \
      --input_file ${HOME}/${dataset}/${SPLIT}.jsonl \
      --output_file ${HOME}/${dataset}/results/${ORDER}/${SEMANTICS}/${window}-${PASSAGE_SIZE}/${model}/${SPLIT}.csv \
      --model_name ${model} --model_token_limit ${window} --passage_len ${PASSAGE_SIZE} --max_new_tokens ${MAX_NEW_TOKENS}
    done
  done
done
