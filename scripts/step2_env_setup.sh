# build environment
uv venv

# start up the venv environment
source .venv/bin/activate.fish

# get dependencies
uv pip install time torch diffusers transformers accelerate Pillow huggingface_hub
