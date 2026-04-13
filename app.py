#!/usr/bin/env python3
"""AI 面试练习 — Gradio 版  python app.py"""

import gradio as gr
import os
import json
from datetime import datetime

from core import make_client, MODEL, COMPANY_STYLES, InterviewSessionBase

client = make_client()

# ─── 用户数据持久化 ────────────────────────────────────────────
_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.json")

def _load_user_data() -> dict:
    try:
        with open(_DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_user_data(**kwargs) -> None:
    data = _load_user_data()
    data.update(kwargs)
    with open(_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── 面试会话 ─────────────────────────────────────────────────
class InterviewSession(InterviewSessionBase):

    def _stream(self, messages: list, system: str):
        stream = client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{"role": "system", "content": system}] + messages,
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                yield ("glm", text)

    def stream_start(self):
        # Phase 1: 匹配预分析
        for tag, chunk in self._stream(
            [{"role": "user", "content": self._analyst_prompt()}],
            self._analyst_sys,
        ):
            yield "analyst", tag, chunk

        # Phase 2: 第一个面试问题
        self._history = [{"role": "user", "content": "请开始面试。"}]
        parts = []
        for tag, chunk in self._stream(self._history, self._interviewer_sys):
            parts.append(chunk)
            yield "interviewer", tag, chunk
        question = "".join(parts)
        self._history.append({"role": "assistant", "content": question})
        self._current_question = question

    def stream_answer(self, answer: str):
        self.round += 1
        self._history.append({"role": "user", "content": answer})
        ctx = self._build_interview_context()
        eval_parts, better_parts, next_parts = [], [], []

        for provider, chunk in self._stream(
            [{"role": "user", "content": f"{ctx}\n\n请对候选人最后一轮的回答进行评估。"}],
            self._evaluator_sys,
        ):
            eval_parts.append(chunk)
            yield "evaluator", provider, chunk

        for provider, chunk in self._stream(
            [{"role": "user", "content": f"{ctx}\n\n请针对最后一轮面试官的问题，提供一个更优质的回答示例（候选人原始回答仅供参考，请勿直接重复）。"}],
            self._better_sys,
        ):
            better_parts.append(chunk)
            yield "better", provider, chunk

        self._rounds.append({
            "question": self._current_question,
            "answer": answer,
            "evaluation": "".join(eval_parts),
            "better": "".join(better_parts),
        })

        self._trim_history()

        for provider, chunk in self._stream(self._history, self._interviewer_sys):
            next_parts.append(chunk)
            yield "interviewer", provider, chunk

        next_q = "".join(next_parts)
        self._history.append({"role": "assistant", "content": next_q})
        self._current_question = next_q


# ─── 辅助：生成右侧反馈 markdown ─────────────────────────────
def feedback_content(eval_text: str, better_text: str) -> str:
    parts = []
    if eval_text:
        parts.append(f"**📊 评估师点评**\n\n{eval_text}")
    if better_text:
        parts.append(f"**⭐ 示范回答**\n\n{better_text}")
    return "\n\n---\n\n".join(parts)


# ─── 事件处理 ────────────────────────────────────────────────
# start_interview outputs: setup_group, interview_section, interview_chat,
#   answer_input, submit_btn, export_btn, session_state, rounds_state, feedback_md, round_selector
START_OUT_N = 10

def start_interview(jd_val, resume_val, style_choice, custom_style_val):
    if not jd_val.strip() or not resume_val.strip():
        gr.Warning("请填写岗位JD和简历")
        return

    style = custom_style_val.strip() if style_choice == "自定义" else COMPANY_STYLES.get(style_choice, "")
    _save_user_data(jd=jd_val, resume=resume_val)
    session = InterviewSession(jd_val, resume_val, style)
    chat = []

    # 切换界面
    yield (
        gr.update(visible=False),           # setup_group
        gr.update(visible=True),            # interview_section
        [],                                 # interview_chat
        gr.update(interactive=False),       # answer_input
        gr.update(interactive=False),       # submit_btn
        gr.update(visible=True),            # export_btn
        session,                            # session_state
        [],                                 # rounds_state
        "",                                 # feedback_md
        gr.update(choices=[], visible=False),  # round_selector
    )

    # 流式输出：预分析 → 第一个面试问题
    current_agent = None
    for agent, provider, chunk in session.stream_start():
        if agent != current_agent:
            current_agent = agent
            if agent == "analyst":
                chat.append({"role": "assistant", "content": "🔍 匹配分析\n\n"})
            elif agent == "interviewer":
                chat.append({"role": "assistant", "content": "第 1 轮\n\n"})
        chat[-1]["content"] += chunk
        yield (gr.update(), gr.update(), list(chat),
               gr.update(), gr.update(), gr.update(),
               session, [], "", gr.update())

    yield (gr.update(), gr.update(), list(chat),
           gr.update(interactive=True), gr.update(interactive=True), gr.update(),
           session, [], "", gr.update())


# submit_answer outputs: interview_chat, answer_input, submit_btn,
#   feedback_md, round_selector, rounds_state, session_state
def submit_answer(answer_val, chat, session, rounds_data):
    if not answer_val.strip() or session is None:
        return

    chat = list(chat)
    chat.append({"role": "user", "content": answer_val})
    yield (list(chat), gr.update(value="", interactive=False),
           gr.update(interactive=False), "", gr.update(), rounds_data, session)

    eval_text, better_text = "", ""
    next_q_started = False
    round_n = 0
    round_choices = []

    for agent, provider, chunk in session.stream_answer(answer_val):
        if agent == "evaluator":
            eval_text += chunk
            fb = feedback_content(eval_text, "")
        elif agent == "better":
            better_text += chunk
            fb = feedback_content(eval_text, better_text)
        else:  # interviewer (next question)
            if not next_q_started:
                next_q_started = True
                # 本轮结束，保存并更新轮次选择器
                rounds_data = rounds_data + [{"evaluation": eval_text, "better": better_text}]
                round_n = len(rounds_data)
                round_choices = [f"第 {i+1} 轮" for i in range(round_n)]
                chat.append({"role": "assistant", "content": f"第 {round_n + 1} 轮\n\n"})
            chat[-1]["content"] += chunk
            fb = feedback_content(eval_text, better_text)

        yield (
            list(chat),
            gr.update(value="", interactive=False),
            gr.update(interactive=False),
            fb,
            gr.update(choices=round_choices, value=f"第 {round_n} 轮" if round_n else None, visible=bool(round_n)),
            rounds_data,
            session,
        )

    yield (
        list(chat),
        gr.update(value="", interactive=True),
        gr.update(interactive=True),
        feedback_content(eval_text, better_text),
        gr.update(choices=round_choices, value=f"第 {round_n} 轮" if round_n else None, visible=bool(round_n)),
        rounds_data,
        session,
    )


def switch_round(round_label, rounds_data):
    if not round_label or not rounds_data:
        return ""
    n = int(round_label.replace("第 ", "").replace(" 轮", "")) - 1
    r = rounds_data[n]
    return feedback_content(r["evaluation"], r["better"])


def export_fn(session):
    if session is None or not session._rounds:
        gr.Warning("暂无面试记录")
        return None
    content = session.export_markdown()
    filename = f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


# ─── UI ──────────────────────────────────────────────────────
css = """
/* === Claude-inspired theme === */
body, .gradio-container { background: #faf9f5 !important; }
footer { display: none !important; }

/* Panel labels */
.panel-label {
  font-size: 0.7rem !important; font-weight: 700 !important;
  letter-spacing: 0.07em !important; text-transform: uppercase !important;
  color: #9b9490 !important; margin-bottom: 8px !important;
}

/* Primary button → terracotta */
button.primary {
  background: #cc785c !important; border-color: #cc785c !important;
  color: #fff !important; border-radius: 8px !important; font-weight: 500 !important;
}
button.primary:hover { background: #b5623f !important; border-color: #b5623f !important; }

/* Secondary button */
button.secondary {
  border-radius: 8px !important; border-color: #e5e0d8 !important; color: #4a4440 !important;
}
button.secondary:hover { background: #f5f0e8 !important; }

/* Inputs */
textarea, .block textarea { background: #fff !important; border-color: #e5e0d8 !important; border-radius: 8px !important; }
textarea:focus, .block textarea:focus { border-color: #cc785c !important; }

/* ── Chatbot ── */
/* Hide all action/copy/share buttons */
.message-buttons, .copy-text-button,
button[title="Copy"], button[title="Share"], button[title="Delete"],
button[title="Like"], button[title="Dislike"],
[data-testid="copy-all"] { display: none !important; }

/* Outer container: white card */
.chatbot {
  background: #fff !important;
  border: 1px solid #e5e0d8 !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}

/* Individual messages: flat, compact, no nested box */
.message > div, .message.bot > div, .message.user > div {
  background: transparent !important;
  border: none !important; border-radius: 0 !important;
  box-shadow: none !important; padding: 0 !important; margin: 0 !important;
}
/* Padding on the row itself — must come after the combined rule above */
.message, .message.bot, .message.user {
  background: transparent !important;
  border: none !important; border-radius: 0 !important; box-shadow: none !important;
  padding: 10px 14px !important;
  border-bottom: 1px solid #f0ece4 !important;
}
.message:last-child { border-bottom: none !important; }
.message.user, .message.user > div { background: #faf7f4 !important; }
/* Unify font size with feedback panel — target text nodes, not structural divs */
.chatbot p, .chatbot li, .chatbot strong, .chatbot em,
.chatbot h1, .chatbot h2, .chatbot h3, .chatbot span {
  font-size: 0.875rem !important;
  line-height: 1.7 !important;
  font-family: inherit !important;
  color: #1a1a1a !important;
}

/* ── Feedback panel: white card matching chatbot ── */
.feedback-md {
  background: #fff !important;
  border: 1px solid #e5e0d8 !important;
  border-radius: 10px !important;
  padding: 16px 18px !important;
  font-size: 0.875rem !important;
  line-height: 1.75 !important;
}

/* ── Style-selector Radio → pill buttons ── */
.style-radio .wrap {
  display: flex !important; flex-wrap: wrap !important; gap: 7px !important;
  padding: 4px 0 !important; background: transparent !important; border: none !important;
}
.style-radio label {
  display: inline-flex !important; align-items: center !important;
  border: 1.5px solid #e5e0d8 !important; border-radius: 20px !important;
  padding: 5px 15px !important; background: #fff !important;
  color: #5a5450 !important; font-size: 0.83rem !important; font-weight: 500 !important;
  cursor: pointer !important; transition: all 0.15s !important; gap: 0 !important;
}
.style-radio label:hover { border-color: #c0b8b0 !important; color: #1a1a1a !important; }
.style-radio label.selected {
  border-color: #cc785c !important; background: #fdf4f0 !important;
  color: #cc785c !important; font-weight: 600 !important;
}
.style-radio input[type="radio"] {
  width: 0 !important; height: 0 !important; opacity: 0 !important; position: absolute !important;
}

/* Export button: text-link style, distinct from primary send */
.export-btn button {
  background: transparent !important;
  border: 1px solid #e5e0d8 !important;
  color: #9b9490 !important;
  font-size: 0.78rem !important;
  font-weight: 400 !important;
  border-radius: 6px !important;
}
.export-btn button:hover { border-color: #cc785c !important; color: #cc785c !important; }
"""

with gr.Blocks(title="AI 面试练习") as demo:
    session_state = gr.State(None)
    rounds_state = gr.State([])

    # ── 设置区 ────────────────────────────────────────────────
    with gr.Group() as setup_group:
        with gr.Accordion("📋 简历脱敏 Prompt（点击展开复制）", open=False):
            gr.Textbox(
                value=(
                    "请帮我对以下内容进行脱敏处理，要求：\n"
                    "- 将真实姓名替换为「候选人」\n"
                    "- 将公司名替换为行业描述，如「某大型互联网公司」\n"
                    "- 将产品/项目名替换为「某项目」\n"
                    "- 将学校名替换为「某高校」\n"
                    "- 保持原有结构、数字指标和工作描述不变\n\n"
                    "需要脱敏的内容：\n[粘贴你的简历]"
                ),
                lines=10, show_label=False, interactive=False,
            )
        _saved = _load_user_data()
        jd_input = gr.Textbox(label="岗位 JD", lines=6, placeholder="粘贴岗位描述...",
                              value=_saved.get("jd", ""))
        resume_input = gr.Textbox(label="简历", lines=8, placeholder="粘贴简历内容（建议脱敏后粘贴，避免真实姓名、公司名等敏感信息）...",
                                  value=_saved.get("resume", ""))

        # 风格选择
        gr.Markdown("**面试官风格**")
        style_radio = gr.Radio(
            list(COMPANY_STYLES.keys()),
            value=list(COMPANY_STYLES.keys())[0],
            container=False,
            elem_classes=["style-radio"],
        )
        first_key = list(COMPANY_STYLES.keys())[0]
        style_desc = gr.Markdown(
            f"<div style='margin:10px 0 4px;padding:12px 16px;background:#faf9f5;border-left:3px solid #cc785c;border-radius:6px;color:#3a3330;font-size:0.88em;line-height:1.7;'>"
            f"<strong style='color:#cc785c;display:block;margin-bottom:4px;'>{first_key}</strong>"
            f"{COMPANY_STYLES[first_key].replace(chr(10), '<br>')}"
            f"</div>"
        )
        custom_style_input = gr.Textbox(
            label="自定义面试官风格",
            lines=4,
            visible=False,
            placeholder="请描述该面试官的特点与偏好，例如：关注哪些方面、追问风格是怎样的、看重什么样的能力...",
        )
        start_btn = gr.Button("开始面试", variant="primary", size="lg")

    # ── 面试区（左右分栏） ─────────────────────────────────────
    with gr.Row(visible=False) as interview_section:

        # 左：对话
        with gr.Column(scale=11, min_width=360):
            gr.HTML("<div class='panel-label'>面试对话</div>")
            interview_chat = gr.Chatbot(
                height=520, label="", layout="messages", show_label=False,
            )
            with gr.Row():
                answer_input = gr.Textbox(
                    placeholder="输入你的回答...", lines=3, label="",
                    scale=5, interactive=False, show_label=False,
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1, interactive=False, min_width=80)
            with gr.Row():
                export_btn = gr.Button("导出面试记录", variant="secondary", visible=False,
                                      size="sm", elem_classes=["export-btn"], min_width=120)
                gr.HTML("")  # spacer
                export_file = gr.File(label="下载文件", visible=False, scale=3)

        # 右：反馈
        with gr.Column(scale=9, min_width=300):
            gr.HTML("<div class='panel-label'>评估 & 示范</div>")
            round_selector = gr.Radio(
                choices=[], label="", visible=False,
                interactive=True, container=False,
                elem_classes=["style-radio"],
            )
            feedback_md = gr.Markdown(
                value="*完成第一轮回答后，这里将显示评估和示范...*",
                height=560,
                elem_classes=["feedback-md"],
            )

    # ── 事件绑定 ──────────────────────────────────────────────
    def on_style_change(choice):
        is_custom = (choice == "自定义")
        desc = COMPANY_STYLES.get(choice, "")
        if is_custom:
            desc_html = "<div style='margin:10px 0 4px;padding:12px 16px;background:#faf9f5;border-left:3px solid #e5e0d8;border-radius:6px;color:#7a7472;font-size:0.88em;font-style:italic;'>请在下方输入自定义面试官风格</div>"
        else:
            desc_html = (
                f"<div style='margin:10px 0 4px;padding:12px 16px;background:#faf9f5;border-left:3px solid #cc785c;"
                f"border-radius:6px;color:#3a3330;font-size:0.88em;line-height:1.7;'>"
                f"<strong style='color:#cc785c;display:block;margin-bottom:4px;'>{choice}</strong>"
                f"{desc.replace(chr(10), '<br>')}</div>"
            )
        return (
            gr.update(visible=is_custom),  # custom_style_input
            gr.update(value=desc_html),    # style_desc
        )

    style_radio.change(
        fn=on_style_change,
        inputs=style_radio,
        outputs=[custom_style_input, style_desc],
    )

    START_OUTPUTS = [
        setup_group, interview_section, interview_chat,
        answer_input, submit_btn, export_btn,
        session_state, rounds_state, feedback_md, round_selector,
    ]
    start_btn.click(fn=start_interview,
                    inputs=[jd_input, resume_input, style_radio, custom_style_input],
                    outputs=START_OUTPUTS)

    ANSWER_OUTPUTS = [interview_chat, answer_input, submit_btn, feedback_md,
                      round_selector, rounds_state, session_state]
    submit_btn.click(fn=submit_answer,
                     inputs=[answer_input, interview_chat, session_state, rounds_state],
                     outputs=ANSWER_OUTPUTS)

    round_selector.change(fn=switch_round,
                          inputs=[round_selector, rounds_state],
                          outputs=[feedback_md])

    export_btn.click(fn=export_fn, inputs=[session_state], outputs=[export_file]).then(
        fn=lambda: gr.update(visible=True), outputs=[export_file]
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1", server_port=7860, inbrowser=True,
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="stone"),
        css=css,
    )
