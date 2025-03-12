docker run --gpus all \
--shm-size 32g \
--name container-test \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
-p 30000:30000 \
sglang-sanic-rag \
--model_path recoilme/recoilme-gemma-2-9B-v0.4 \
--data_path ./data \
--port 30000 \
--single_process \
--retriever_model sentence-transformers/static-similarity-mrl-multilingual-v1 \
--renew_db