"""
The retrieval knowledge base the self-improving harness writes lessons
into and reads them back from -- the actual mechanism behind "harness can
create its own evals/data, change its own prompts/tools." A lesson here
is plain text (what went wrong, what to do instead), embedded with
nomic-embed-text (confirmed local, 768-dim), retrieved by cosine
similarity against a new task's instruction and injected into the
agent's system prompt as in-context guidance -- this is the ICL-hillclimb
mechanism from the original plan, now wired to something real.
"""
import json
import math
from pathlib import Path

import requests

OLLAMA_EMBED = "http://localhost:11434/api/embeddings"


def embed(text: str) -> list[float]:
    resp = requests.post(OLLAMA_EMBED, json={"model": "nomic-embed-text", "prompt": text}, timeout=30)
    return resp.json()["embedding"]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class LessonKB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lessons: list[dict] = []
        if self.path.exists():
            self.lessons = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def add(self, lesson_text: str, source_task_id: str, subtype: str, scope: str = "task_specific"):
        entry = {"lesson": lesson_text, "source_task_id": source_task_id, "subtype": subtype,
                 "scope": scope, "embedding": embed(lesson_text)}
        self.lessons.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def retrieve(self, query: str, k: int = 3, min_similarity: float = 0.3) -> list[str]:
        """Universal (process/sequencing) lessons are always included --
        confirmed bug: pure similarity ranked one dead last (0.547, lowest
        of six) against a task whose topic shares no vocabulary with it,
        even though "always populate the timeline before rendering" applies
        to literally every task. Topical lessons still go through
        similarity search, since that part works fine for same-subject
        lessons (e.g. two different audio-ducking mistakes)."""
        if not self.lessons:
            return []
        universal = [l["lesson"] for l in self.lessons if l.get("scope") == "universal"]
        topical = [l for l in self.lessons if l.get("scope") != "universal"]
        if not topical:
            return universal
        q_emb = embed(query)
        scored = [(_cosine(q_emb, l["embedding"]), l["lesson"]) for l in topical]
        scored = [s for s in scored if s[0] >= min_similarity]
        scored.sort(key=lambda x: -x[0])
        return universal + [lesson for _, lesson in scored[:k]]
