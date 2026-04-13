#!/usr/bin/env python3
"""AI 面试练习 Web 服务 — python server.py"""

import os
import json
import uuid
import time
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel

from openai import OpenAI
from core import MODEL, DEFAULT_INTERVIEWER_STYLE, InterviewSessionBase

_SESSION_TTL = 2 * 3600  # 2 小时不活跃后清理
_sessions: dict[str, tuple["InterviewSession", float]] = {}


def _cleanup():
    now = time.time()
    expired = [k for k, (_, ts) in list(_sessions.items()) if now - ts > _SESSION_TTL]
    for k in expired:
        del _sessions[k]


def _get_session(session_id: str) -> "InterviewSession | None":
    if session_id not in _sessions:
        return None
    session, _ = _sessions[session_id]
    _sessions[session_id] = (session, time.time())
    return session


# ─── SSE helper ───────────────────────────────────────────────
def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ─── 面试会话 ─────────────────────────────────────────────────
class InterviewSession(InterviewSessionBase):

    def __init__(self, jd: str, resume: str, style: str, api_key: str):
        super().__init__(jd, resume, style)
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )
        self._last_text: str = ""

    def _stream_section(self, messages: list, system: str, agent: str):
        """Yield SSE 事件；完成后将完整文本存入 self._last_text。"""
        yield sse({"type": "section", "agent": agent})
        parts: list[str] = []
        stream = self._client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{"role": "system", "content": system}] + messages,
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                parts.append(text)
                yield sse({"type": "token", "text": text})
        self._last_text = "".join(parts)

    def stream_start(self):
        # Phase 1: 匹配预分析
        yield from self._stream_section(
            [{"role": "user", "content": self._analyst_prompt()}],
            self._analyst_sys, "analyst",
        )

        # Phase 2: 第一个面试问题
        self._history = [{"role": "user", "content": "请开始面试。"}]
        yield from self._stream_section(self._history, self._interviewer_sys, "interviewer")
        self._history.append({"role": "assistant", "content": self._last_text})
        self._current_question = self._last_text
        yield sse({"type": "done"})

    def stream_answer(self, answer: str):
        self.round += 1
        self._history.append({"role": "user", "content": answer})
        ctx = self._build_interview_context()

        yield from self._stream_section(
            [{"role": "user", "content": f"{ctx}\n\n请对候选人最后一轮的回答进行评估。"}],
            self._evaluator_sys, "evaluator",
        )
        evaluation = self._last_text

        yield from self._stream_section(
            [{"role": "user", "content": f"{ctx}\n\n请针对最后一轮面试官的问题，提供一个更优质的回答示例（候选人原始回答仅供参考，请勿直接重复）。"}],
            self._better_sys, "better",
        )
        better = self._last_text

        self._rounds.append({
            "question": self._current_question,
            "answer": answer,
            "evaluation": evaluation,
            "better": better,
        })

        self._trim_history()

        yield from self._stream_section(self._history, self._interviewer_sys, "interviewer")
        next_q = self._last_text
        self._history.append({"role": "assistant", "content": next_q})
        self._current_question = next_q

        yield sse({"type": "done"})


# ─── FastAPI App ──────────────────────────────────────────────
app = FastAPI()

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(path, encoding="utf-8") as f:
        return f.read()


class StartRequest(BaseModel):
    jd: str
    resume: str
    style: str = ""
    api_key: str = ""


@app.post("/start")
def start(req: StartRequest):
    api_key = req.api_key.strip() or os.environ.get("GLM_API_KEY", "")
    if not api_key:
        return Response("GLM API Key is required", status_code=400)
    _cleanup()
    sid = str(uuid.uuid4())
    style = req.style.strip() or DEFAULT_INTERVIEWER_STYLE
    session = InterviewSession(req.jd, req.resume, style, api_key)
    _sessions[sid] = (session, time.time())
    return StreamingResponse(
        session.stream_start(),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, "X-Session-Id": sid},
    )


class AnswerRequest(BaseModel):
    answer: str
    session_id: str


@app.post("/answer")
def answer_route(req: AnswerRequest):
    session = _get_session(req.session_id)
    if session is None:
        return Response("session not found", status_code=404)
    return StreamingResponse(session.stream_answer(req.answer), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/export")
def export_route(session_id: str = Query(...)):
    session = _get_session(session_id)
    if session is None or not session._rounds:
        return Response("no data", status_code=404)
    content = session.export_markdown()
    filename = f"interview_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    return Response(
        content=content.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
