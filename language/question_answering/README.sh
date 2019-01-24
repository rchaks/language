#!/usr/bin/env bash

set -xeo pipefail

echo "############################"
echo "Set up file locations to use"
echo "############################"

export DATA_DIR=/dccstor/rishavc1/Data/NQ

export EXPERIMENT_DIR=/dccstor/rishavc1/Experiments/NQ/google-baseline
export MODELS_DIR=${EXPERIMENT_DIR}/models

export EMBEDDINGS_DIR=/dccstor/rishavc1/Data/Embeddings
export EMBEDDINGS_FILE=${EMBEDDINGS_DIR}/glove.840B.300d.txt

echo "#######################"
echo "Step 1.0: Data Download"
echo "#######################"

export NQ_DATA_DIR=${DATA_DIR}/natural_questions/v1.0

if [ -d "${NQ_DATA_DIR}" ]; then
    echo "Skipping download of underlying data set since assuming it's been done already"
else
    mkdir -p ${DATA_DIR}
    gsutil -m cp -r gs://natural_questions ${DATA_DIR}
fi

echo "#########################"
echo "Step 2.0: Preprocess Data"
echo "#########################"

if ls ${NQ_DATA_DIR}/train/nq-train-*.short_pipeline.tfr 1> /dev/null 2>&1; then
    echo "Skipping preprocessing of train files for short answer"
else
    python -m language.question_answering.preprocessing.create_nq_short_pipeline_examples \
    --input_pattern=${NQ_DATA_DIR}/train/nq-train-*.jsonl.gz \
    --output_dir=${NQ_DATA_DIR}/train
fi

if ls ${NQ_DATA_DIR}/dev/nq-dev-*.short_pipeline.tfr 1> /dev/null 2>&1; then
    echo "Skipping preprocessing of dev files for short answer"
else
    python -m language.question_answering.preprocessing.create_nq_short_pipeline_examples \
    --input_pattern=${NQ_DATA_DIR}/dev/nq-dev-*.jsonl.gz \
    --output_dir=${NQ_DATA_DIR}/dev
fi

## TODO: Long Answer
#python -m language.question_answering.preprocessing.create_nq_long_examples \
#  --input_pattern=${NQ_DATA_DIR}/dev/nq-dev-*.jsonl.gz \
#  --output_dir=${NQ_DATA_DIR}/dev
#
#python -m language.question_answering.preprocessing.create_nq_long_examples \
#  --input_pattern=${NQ_DATA_DIR}/train/nq-train-*.jsonl.gz \
#  --output_dir=${NQ_DATA_DIR}/train

echo "#############################"
echo "Step 3.0: Download embeddings"
echo "#############################"

if [ -f "${EMBEDDINGS_FILE}" ]; then
    echo "Skipping download of ${EMBEDDINGS_FILE} since it looks like it was already downloaded"
else
    mkdir -p ${EMBEDDINGS_DIR}
    curl https://nlp.stanford.edu/data/glove.840B.300d.zip > ${EMBEDDINGS_DIR}/glove.840B.300d.zip
    unzip ${EMBEDDINGS_DIR}/glove.840B.300d.zip -d ${EMBEDDINGS_DIR}
fi

echo "##########################################"
echo "Step 4.0: Short Answer Model Training     "
echo "##########################################"

python -m language.question_answering.experiments.nq_short_pipeline_experiment \
  --embeddings_path=${EMBEDDINGS_FILE} \
  --nq_short_pipeline_train_pattern=${NQ_DATA_DIR}/train/nq-train-*.short_pipeline.tfr \
  --nq_short_pipeline_eval_pattern=${NQ_DATA_DIR}/dev/nq-dev-*.short_pipeline.tfr \
  --num_eval_steps=10 \
  --model_dir=${MODELS_DIR}/nq_short_pipeline

# TODO
#echo "##########################################"
#echo "Step 5.0: Short Answer Model Training     "
#echo "##########################################"

