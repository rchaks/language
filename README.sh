#!/usr/bin/env bash

set -xeo pipefail

echo "############################"
echo "Set up file locations to use"
echo "############################"

# Mocks just for runnning with sample data
#export DATA_DIR="/Users/chakravr/Data/NaturalQuestions"
#export EXPERIMENT_DIR="/Users/chakravr/Experiments/NaturalQuestions/Google-Base-OOB/sample-data-only"
#export EMBEDDINGS_DIR="/Users/chakravr/Data/WordEmbeddings/"
#export EMBEDDINGS_FILE=${EMBEDDINGS_DIR}/enwiki_yhans_voc20_400wv.ctw9.txt
#export NQ_DATA_DIR=${DATA_DIR}/sample_questions/v1.0

# Real paths
export DATA_DIR=/dccstor/rishavc1/Data/NQ
export EXPERIMENT_DIR=/dccstor/rishavc1/Experiments/NQ/google-baseline
export EMBEDDINGS_DIR=/dccstor/rishavc1/Data/Embeddings
export EMBEDDINGS_FILE=${EMBEDDINGS_DIR}/glove.840B.300d.txt
export NQ_DATA_DIR=${DATA_DIR}/natural_questions/v1.0


export MODELS_DIR=${EXPERIMENT_DIR}/models

echo "#######################"
echo "Step 1.0: Data Download"
echo "#######################"

if [ -d "${NQ_DATA_DIR}" ]; then
    echo "Skipping download of underlying data set since assuming it's been done already"
else
    mkdir -p ${DATA_DIR}
    gsutil -m cp -r gs://natural_questions ${DATA_DIR}
fi

echo "########################################"
echo "Step 2.1: Preprocess Data (LONG ANSWER)"
echo "########################################"


if ls ${NQ_DATA_DIR}/train/nq-train-*.long.tfr 1> /dev/null 2>&1; then
    echo "Skipping preprocessing of train files for LONG answer"
else
    python -m language.question_answering.preprocessing.create_nq_long_examples \
    --input_pattern=${NQ_DATA_DIR}/train/nq-train-*.jsonl.gz \
    --output_dir=${NQ_DATA_DIR}/train
fi

if ls ${NQ_DATA_DIR}/dev/nq-dev-*.long.tfr 1> /dev/null 2>&1; then
    echo "Skipping preprocessing of dev files for LONG answer"
else
    python -m language.question_answering.preprocessing.create_nq_long_examples \
    --input_pattern=${NQ_DATA_DIR}/dev/nq-dev-*.jsonl.gz \
    --output_dir=${NQ_DATA_DIR}/dev
fi

echo "########################################"
echo "Step 2.2: Preprocess Data (SHORT ANSWER)"
echo "########################################"


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
echo "Step 4.0: Long Answer Model Training     "
echo "##########################################"

python -m language.question_answering.experiments.nq_long_experiment \
  --embeddings_path=${EMBEDDINGS_FILE} \
  --nq_long_train_pattern=${NQ_DATA_DIR}/train/nq-train-*.long.tfr \
  --nq_long_eval_pattern=${NQ_DATA_DIR}/dev/nq-dev-*.long.tfr \
  --num_eval_steps=100 \
  --batch_size=4 \
  --model_dir=${MODELS_DIR}/nq_long

echo "##########################################"
echo "Step 5.0: Short Answer Model Training     "
echo "##########################################"

python -m language.question_answering.experiments.nq_short_pipeline_experiment \
  --embeddings_path=${EMBEDDINGS_FILE} \
  --nq_short_pipeline_train_pattern=${NQ_DATA_DIR}/train/nq-train-*.short_pipeline.tfr \
  --nq_short_pipeline_eval_pattern=${NQ_DATA_DIR}/dev/nq-dev-*.short_pipeline.tfr \
  --num_eval_steps=10 \
  --model_dir=${MODELS_DIR}/nq_short_pipeline