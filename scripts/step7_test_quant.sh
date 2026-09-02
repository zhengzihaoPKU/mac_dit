uv run python src/run_dit.py \
    --load-quantized ./model/quantized/DiT-XL-2-256-int4-g128 \
    --quant-backend metal