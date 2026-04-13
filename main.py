#!/usr/bin/env python3
"""AI 面试练习多Agent系统（命令行版）"""

import os
from datetime import datetime

from core import make_client, MODEL, DEFAULT_INTERVIEWER_STYLE, InterviewSessionBase

client = make_client()

SEP      = "─" * 58
SEP_THIN = "╌" * 58


# ─── 终端流式输出 ──────────────────────────────────────────────
def _stream_print(messages: list, system: str) -> str:
    chunks = []
    stream = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "system", "content": system}] + messages,
        stream=True,
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content or ""
        if text:
            print(text, end="", flush=True)
            chunks.append(text)
    print()
    return "".join(chunks)


# ─── Session ───────────────────────────────────────────────────
class InterviewSession(InterviewSessionBase):

    def analyze(self) -> None:
        """面试前：简历与JD匹配预分析。"""
        _print_header("🔍  匹配分析")
        _stream_print(
            [{"role": "user", "content": self._analyst_prompt()}],
            self._analyst_sys,
        )

    def start(self) -> str:
        _print_header("🎤  面试官")
        self._history = [{"role": "user", "content": "请开始面试。"}]
        question = _stream_print(self._history, self._interviewer_sys)
        self._history.append({"role": "assistant", "content": question})
        self._current_question = question
        return question

    def process_answer(self, answer: str) -> None:
        self.round += 1
        self._history.append({"role": "user", "content": answer})
        ctx = self._build_interview_context()

        _print_header("📊  评估师")
        evaluation = _stream_print(
            [{"role": "user", "content": f"{ctx}\n\n请对候选人最后一轮的回答进行评估。"}],
            self._evaluator_sys,
        )

        _print_header("⭐  示范回答（同等简历背景下更优的表达）")
        better = _stream_print(
            [{"role": "user", "content": f"{ctx}\n\n请针对最后一轮面试官的问题，提供一个更优质的回答示例（候选人原始回答仅供参考，请勿直接重复）。"}],
            self._better_sys,
        )

        self._rounds.append({
            "question": self._current_question,
            "answer": answer,
            "evaluation": evaluation,
            "better": better,
        })

        self._trim_history()

        _print_header("🎤  面试官（继续）")
        next_q = _stream_print(self._history, self._interviewer_sys)
        self._history.append({"role": "assistant", "content": next_q})
        self._current_question = next_q


# ─── UI helpers ────────────────────────────────────────────────
def _print_header(label: str) -> None:
    print(f"\n{SEP}\n{label}\n{SEP_THIN}")


def _get_input(prompt: str, multiline: bool = False) -> str:
    if not multiline:
        return input(prompt).strip()
    print(prompt)
    print("  （多行输入；输完后单独一行输入 '---' 确认）")
    lines = []
    while True:
        line = input()
        if line.strip() == "---":
            break
        lines.append(line)
    return "\n".join(lines)


# ─── Main ──────────────────────────────────────────────────────
def main() -> None:
    print("=" * 58)
    print("        AI 面试练习系统（三Agent协作）")
    print("=" * 58)
    print("\n  面试官提问 → 你回答 → 评估师点评 → 示范更优回答\n")

    jd     = _get_input("📋 岗位JD（粘贴后输入 '---'）：\n", multiline=True)
    print()
    resume = _get_input("📄 你的简历（粘贴后输入 '---'）：\n", multiline=True)
    print()

    print("🎭 面试官风格：")
    print("  1. 默认（专业技术/业务面试官，注重项目经验与细节追问）")
    print("  2. 自定义")
    choice = input("请选择（1/2，回车默认1）：").strip() or "1"
    if choice == "2":
        style = _get_input("\n请描述面试官的特点与偏好（输入 '---' 结束）：\n", multiline=True)
    else:
        style = DEFAULT_INTERVIEWER_STYLE
        print("\n✓ 已使用默认风格")

    session = InterviewSession(jd, resume, style)
    session.analyze()

    print(f"\n{'=' * 58}\n  面试开始！随时输入 'quit' 退出。\n{'=' * 58}")
    session.start()

    while True:
        print(f"\n{SEP_THIN}")
        user_input = _get_input("💬 你的回答（多行请输入 '---' 结束）：\n", multiline=True)

        if user_input.lower() in ("quit", "exit", "退出", "q"):
            print(f"\n{'=' * 58}\n  面试练习结束！祝面试顺利 🎉\n{'=' * 58}")
            if session.round > 0:
                filename = f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(session.export_markdown())
                print(f"\n  📄 记录已导出：{filepath}")
            break

        if not user_input:
            print("  请输入回答，或输入 quit 退出。")
            continue

        session.process_answer(user_input)


if __name__ == "__main__":
    main()
