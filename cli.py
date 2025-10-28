import argparse
import sys
from typing import Optional

from .llm_loader import load_llm
from .verifier import verify


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Code Verification & Feedback Agent (LangChain + Hugging Face)",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--code", type=str, help="Code string to verify")
    src.add_argument("--code-file", type=str, help="Path to a code file to verify")

    parser.add_argument("--task", type=str, required=True, help="Natural-language task")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model repo/name (HF Hub or local). Defaults to FLAN-T5 base",
    )
    parser.add_argument(
        "--hub",
        action="store_true",
        help="Use Hugging Face Hub backend if available",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--detailed", action="store_true", help="Provide line-specific guidance when verdict is No")

    args = parser.parse_args(argv)

    code = args.code if args.code is not None else read_text_file(args.code_file)

    llm = load_llm(
        model=args.model,
        use_hf_hub=args.hub,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    result = verify(task=args.task, code=code, llm=llm, detailed=args.detailed)

    if result.is_satisfied:
        print("Yes")
        return 0

    print(result.feedback or "No: Missing feedback")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
