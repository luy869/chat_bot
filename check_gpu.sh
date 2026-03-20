#!/bin/bash

echo "=== Ollama GPU Status ==="
docker exec test-rag-platform-ollama sh -c 'echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" && echo "OLLAMA_NUM_PARALLEL: $OLLAMA_NUM_PARALLEL"'

echo ""
echo "=== Docker GPU Allocation ==="
docker inspect test-rag-platform-ollama | grep -A 5 "Devices"

echo ""
echo "=== Ollama Logs (GPU usage) ==="
docker logs test-rag-platform-ollama | grep -i "gpu\|cuda" | tail -10
