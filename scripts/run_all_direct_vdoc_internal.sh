export ES_HOST="https://convaidp-nlp.sl.cloud9.ibm.com:9200"
export ES_AUTH="elastic:w6Vm=nH8w=pDegeNCx5K"
export ES_INDEX="scrolls-sliding-window-elser-2705"
#export ES_INDEX="scrolls-narrativeqa-val500-sliding-window-elser-020624"
#export ES_INDEX="test-googlenq-v1.0-test-sliding-window-elser0906"
export ES_API_KEY="UjhNeFI1RUJ2OWJkWWFCbVR5aTk6RG5EeTlkVWVRS0NjekVHb2p4bi03QQ=="
export ES_SSL_FINGERPRINT="8dfd0396c40b1591603b32d4b0e64dfe402432dd0029fe1d93afb39b3dd5ce9d"

#HOME=/dccstor/yosimass1/resources/scrolls
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
      #export ES_INDEX=zero-scrolls-${dataset}-test-sliding-window-elser-0406
      PYTHONPATH=. python -u vdoc.py \
      --dataset scrolls \
      --cache_id ${SPLIT}/${dataset} \
      --order ${ORDER} \
      --input_file ${HOME}/${dataset}/${SPLIT}.jsonl \
      --output_file ${HOME}/${dataset}/results/${ORDER}/${SEMANTICS}/${window}-${PASSAGE_SIZE}/${model}/${SPLIT}.csv \
      --model_name ${model} --model_token_limit ${window} --passage_len ${PASSAGE_SIZE} --max_new_tokens ${MAX_NEW_TOKENS}
    done
  done
done
