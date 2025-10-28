from dataclasses import dataclass
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


MAX_PROMPT_CHARS = 900
TRUNCATION_NOTICE = "\n...\n[truncated for local model context]\n"


EVALUATION_PROMPT_SIMPLE = """
You are a strict code verification agent.
Given a natural-language Task and a candidate code Result, decide if the Result fully satisfies the Task.
Output exactly one line:
- "Yes" if it fully satisfies the Task.
- "No: <clear, actionable feedback>" otherwise.
Be concise but specific. Do not add extra lines, bullets, or explanations after your single line.

Task:
{task}

Result (code):
```python
{code}
```

Your single-line verdict:
""".strip()


EVALUATION_PROMPT_DETAILED = """
You are a strict code verification agent.
Given a natural-language Task and a candidate code Result, decide if the Result fully satisfies the Task.

Output format requirements:
- First line MUST be exactly either "Yes" or "No: <brief reason>".
- If the first line is "No: ...", then provide a single paragraph summary of all essential changes needed:
  - List the key issues and how to fix them in one cohesive paragraph
  - Be specific about what needs to change and why
  - Focus on the most critical problems that prevent the code from satisfying the task

Task:
{task}

Code with line numbers:
```python
{numbered_code}
```

Your verdict and, if No, essential changes summary:
""".strip()


def build_verifier_chain(llm, detailed: bool = False):
    if detailed:
        template = PromptTemplate.from_template(EVALUATION_PROMPT_DETAILED)
    else:
        template = PromptTemplate.from_template(EVALUATION_PROMPT_SIMPLE)
    return template | llm | StrOutputParser()


@dataclass
class VerificationResult:
    is_satisfied: bool
    feedback: Optional[str]


def parse_verdict(verdict: str) -> VerificationResult:
    if verdict is None:
        return VerificationResult(False, "No: Empty model output")

    text = verdict.strip()
    first_line = text.splitlines()[0].strip() if text else ""

    if first_line.lower().startswith("yes"):
        return VerificationResult(True, None)

    if first_line.lower().startswith("no"):
        return VerificationResult(False, text if text else "No: Unknown issue")

    return VerificationResult(False, f"No: Follow the required format. Model said: {text}")


def _add_line_numbers(code: str) -> str:
    lines = code.splitlines()
    return "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))


def _truncate_for_local_model(text: str) -> str:
    if len(text) <= MAX_PROMPT_CHARS:
        return text

    available = MAX_PROMPT_CHARS - len(TRUNCATION_NOTICE)
    if available <= 0:
        return TRUNCATION_NOTICE.strip()

    return text[:available] + TRUNCATION_NOTICE


def verify(task: str, code: str, llm, detailed: bool = False) -> VerificationResult:
    chain = build_verifier_chain(llm, detailed=detailed)
    if detailed:
        numbered_code = _add_line_numbers(code)
        prompt_code = _truncate_for_local_model(numbered_code)
        verdict = chain.invoke({"task": task, "numbered_code": prompt_code})
    else:
        prompt_code = _truncate_for_local_model(code)
        verdict = chain.invoke({"task": task, "code": prompt_code})
    return parse_verdict(verdict)
