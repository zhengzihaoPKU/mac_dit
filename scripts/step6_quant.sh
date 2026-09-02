uv run python src/run_dit.py \
    --quantization int4 \
    --group-size 128 \
    --quant-backend reference \
    --quantize-only
