### Code Verification & Feedback Agent (LangChain + Hugging Face)

A small, runnable Python project that evaluates whether a given code "Result" fully satisfies a natural-language "Task". The agent returns:

- **Yes**: if the code fully satisfies the task
- **No: <actionable feedback>**: if it does not, with clear guidance on what is missing or incorrect

This project uses LangChain with either Hugging Face Hub or a local Transformers pipeline.

---

### Quickstart

1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) (Optional) Set Hugging Face Hub token for hosted inference

```bash
export HUGGINGFACEHUB_API_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

4) Run the verifier via CLI

- Using a code file:

```bash
python -m agent_verification.cli \
  --task "Write a function add(a, b) that returns a+b." \
  --code-file examples/sample_code.py
```

- Or passing code directly:

```bash
python -m agent_verification.cli \
  --task "Write a function add(a, b) that returns a+b." \
  --code "def add(a, b):\n    return a + b"
```

- For local inference (no HF Hub), the default model is the Ollama tag `llama3:latest` and runs via `transformers`. If you prefer or need HF Hub, pass `--hub` or set the token.

```bash
python -m agent_verification.cli --task "..." --code-file path/to/code.py --hub
```

---

### Notes on Models

- By default, the agent chooses a sensible small model for CPU:
  - Local Ollama: `llama3:latest`
  - HF Hub (if `--hub` or token present): `google/flan-t5-base`
- You can override with `--model <repo_or_local_name>`.
- Local inference for larger models requires adequate CPU/GPU and installed backends (`torch`).

---

### Output Contract

- Exactly one line starting with either `Yes` or `No:`
  - `Yes` means the provided code fully satisfies the task requirements
  - `No: ...` includes clear and actionable feedback on what to change

---

### Project Layout

- `agent_verification/llm_loader.py`: LLM loader for HF Hub or local pipeline
- `agent_verification/verifier.py`: Prompt, chain, and output parsing
- `agent_verification/cli.py`: CLI entrypoint
- `examples/`: Example task/code files

---

### Troubleshooting

- If local pipeline fails due to missing backends, ensure `torch` is installed for your platform. You may need specific install commands from PyTorch docs.
- If HF Hub calls fail, ensure `HUGGINGFACEHUB_API_TOKEN` is set and the model repo supports text generation for your request.

---

### License

MIT


#### Detailed feedback

Use `--detailed` to get a comprehensive summary of essential changes when the answer is No. Example:

```bash
python -m agent_verification.cli \
  --task "Write a function add(a, b) that returns a+b." \
  --code-file examples/bad_code.py \
  --detailed
```

Output format:
- First line: `No: ...`
- Followed by a single paragraph summarizing all essential changes needed
