import os
import re
from typing import List, Dict


def _parse_md_conversation(path: str) -> List[Dict[str, str]]:
    """Parse a markdown conversation file into a list of (role, content) dicts.

    Expects the files to mark turns with '**User**' and '**Agent**'.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    parts = re.split(r"\n### Turn \d+\n", text)
    messages = []
    for part in parts:
        # find User and Agent blocks
        user_match = re.search(r"\*\*User\*\*\n\n> (.+?)(?:\n\n|$)", part, re.S)
        agent_match = re.search(r"\*\*Agent\*\*\n\n(.+?)(?:\n\n|$)", part, re.S)
        if user_match and agent_match:
            user_text = user_match.group(1).strip().replace("\n", " ")
            agent_text = agent_match.group(1).strip().replace("\n", " ")
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": agent_text})

    return messages


def load_few_shot_examples(max_examples: int = 5) -> List[Dict[str, str]]:
    """Load up to `max_examples` few-shot examples from the extracted sample folder.

    Returns a flat list of messages (user/assistant pairs) suitable for prepending
    to a chat conversation.
    """
    base = os.path.join(os.getcwd(), "_sample_conversations", "GenAI_SampleConversations")
    if not os.path.isdir(base):
        return []

    files = sorted([os.path.join(base, f) for f in os.listdir(base) if f.endswith(".md")])
    examples = []
    for p in files[:max_examples]:
        msgs = _parse_md_conversation(p)
        # take the first meaningful user+assistant pair from each file
        for m in msgs[:2]:
            examples.append(m)
    return examples
