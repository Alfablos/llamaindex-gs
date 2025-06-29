# LlamaIndex Getting Started

A few demos about LlamaIndex functionalities

1. Agents and Workflows
2. RAGs
3. Structured Data

## Usage

```shell
nix develop

cd 01_concatmath
uv venv --python 3.13       # match the flake.nix
source .venv/bin/activate
uv sync

export OPENAI_API_KEY=...

uv run main.py
```

or

```shell
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export MY_VAR=my_value

nix develop --impure

cd 01_concatmath
uv venv --python 3.13       # match the flake.nix
source .venv/bin/activate
uv sync

uv run main.py
```
