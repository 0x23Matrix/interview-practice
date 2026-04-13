#!/usr/bin/env python3
"""AI 面试练习 — 共享核心模块"""

from openai import OpenAI
import os
from datetime import datetime

# ─── Client & Model ──────────────────────────────────────────
def make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("GLM_API_KEY", ""),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")

# 保留最近 N 轮对话，超出后裁剪，防止 context 溢出
HISTORY_KEEP_ROUNDS = 5

# ─── 面试官风格预设 ────────────────────────────────────────────
DEFAULT_INTERVIEWER_STYLE = """专业严谨的技术/业务面试官，具体特点：
- 注重实际项目经验和可量化的工作成果
- 喜欢追问技术细节、业务逻辑和背后的决策依据
- 欣赏结构清晰、逻辑严密的回答（如STAR法则）
- 重视候选人的学习能力和解决复杂问题的思路
- 对过于笼统、夸大或与简历不符的陈述比较敏感
- 会适时追问薄弱点，考察深度"""

COMPANY_STYLES: dict[str, str] = {
    "默认 · 专业通用": DEFAULT_INTERVIEWER_STYLE,

    "腾讯 · 用户价值": """腾讯产品经理面试官风格：
- 无论项目背景多 tech、多前沿，最后几乎都会被拉回到一个问题："这个功能/产品，对用户的价值到底是什么？用户真的需要吗？"
- 会追问用户体验细节，经常要求评价一个日常使用的不好用 app/功能，并提出具体改进方案
- 考察用户同理心和产品直觉，看重以用户价值为核心的思考方式""",

    "字节 · 深度追问": """字节跳动技术面试官风格：
- 特别喜欢对一个问题深挖到底，例如：为什么选这个技术？考虑过其他方案吗？遇到最难的点在哪？怎么解决的？
- 期待你把每一层逻辑都拆解清楚，场面严肃、被反复 challenge 很常见
- 看重深度思考的习惯、面对追问时的冷静表达，以及结果导向的落地能力""",

    "阿里 · 格局洞察": """阿里巴巴面试官风格：
- 喜欢探讨行业趋势、技术发展方向、产品的战略价值这类偏宏观的问题
- 例如："你怎么看待大模型对当前产品形态的影响？""未来一两年最大的机会点在哪里？"
- 看重独立深刻的行业认知和系统性的思考框架""",

    "自定义": "",
}


# ─── System prompts ────────────────────────────────────────────
def interviewer_system(jd: str, resume: str, style: str) -> str:
    return f"""你是一位正在进行面试的专业面试官。

## 应聘岗位JD
{jd}

## 候选人简历
{resume}

## 你的面试风格与偏好
{style}

## 行为准则
- 每次只问一个问题，等待候选人回答后再继续
- 问题需结合JD要求和候选人简历背景，有针对性
- 可以对感兴趣的点进行深入追问
- 始终保持你设定的面试风格
- 使用中文进行面试
- 开场先做一句话自我介绍，然后直接开始第一个问题
- 不要给候选人提示或评价，保持中立的面试官姿态"""


def evaluator_system(jd: str, resume: str, style: str) -> str:
    return f"""你是一位资深面试培训师，帮助候选人提升面试表现。

## 应聘岗位JD
{jd}

## 候选人简历
{resume}

## 本次面试官的风格与偏好
{style}

## 评估要求
收到面试问题和候选人的回答后，请输出以下结构：

**维度评分**
- 相关性（是否直接回应问题）：X/10
- 结构性（逻辑层次、STAR法则等）：X/10
- 具体性（案例/数据支撑，避免空泛）：X/10
- 岗位匹配度（与JD要求的契合程度）：X/10
- 表达力（简洁有力，适合口头表达）：X/10

**综合评分：X/10**
一句话点评核心问题

**亮点**
- （2-3个做得好的地方）

**不足**
- （2-3个主要弱点或遗漏）

**优化方向**
- （结合面试官偏好，给出具体可操作的改进建议）

语言简洁直接，避免套话，使用中文。"""


def better_candidate_system(jd: str, resume: str, style: str) -> str:
    return f"""你是与候选人拥有相同简历背景的"更优秀版本"，帮助候选人看到更好的表达方式。

## 应聘岗位JD
{jd}

## 你的背景（与候选人完全相同的简历）
{resume}

## 面试官风格与偏好
{style}

## 行为准则
- 基于相同的简历经历，展示更优质的回答——不编造简历中没有的经历
- 充分挖掘和展示已有经历的价值，使用STAR法则等结构化表达
- 针对面试官的偏好定制回答重点
- 语言简洁有力，控制在适合口语表达的长度（约200-400字）
- 回答结束后，用一行「【关键改进点】」简要说明相比原始回答做了哪些提升
- 使用中文"""


def analyst_system() -> str:
    return """你是一位专业的求职顾问，在面试开始前对候选人的简历与岗位JD进行匹配分析。

## 分析要求
请输出以下结构，总字数控制在350字以内：

**匹配评分：X/10**
一句话说明整体匹配情况

**简历亮点**（针对JD，2-3条）
- ...

**潜在薄弱点 & 备考建议**（2-3条）
- ...

**面试官可能重点考察的方向**（2-3条）
- ...

语言简洁直接，避免套话，使用中文。"""


# ─── Base session ──────────────────────────────────────────────
class InterviewSessionBase:
    """共享状态与纯逻辑；具体的流式输出由子类实现。"""

    def __init__(self, jd: str, resume: str, style: str):
        self.jd     = jd
        self.resume = resume
        self.style  = style
        self._interviewer_sys = interviewer_system(jd, resume, style)
        self._evaluator_sys   = evaluator_system(jd, resume, style)
        self._better_sys      = better_candidate_system(jd, resume, style)
        self._analyst_sys     = analyst_system()
        self._history: list       = []
        self._current_question    = ""
        self.round: int           = 0
        self._rounds: list[dict]  = []

    def _analyst_prompt(self) -> str:
        return (
            f"## 岗位JD\n{self.jd}\n\n"
            f"## 候选人简历\n{self.resume}\n\n"
            "请进行匹配分析。"
        )

    def _build_interview_context(self) -> str:
        lines = ["## 完整面试对话记录\n"]
        for msg in self._history[1:]:
            role = "面试官" if msg["role"] == "assistant" else "候选人"
            lines.append(f"{role}：{msg['content']}\n")
        return "\n".join(lines)

    def _trim_history(self) -> None:
        """超过 HISTORY_KEEP_ROUNDS 轮时裁剪：保留首条初始消息 + 最近 N 轮 Q&A。"""
        max_msgs = 1 + HISTORY_KEEP_ROUNDS * 2
        if len(self._history) > max_msgs:
            self._history = [self._history[0]] + self._history[-HISTORY_KEEP_ROUNDS * 2:]

    def export_markdown(self) -> str:
        lines = [
            "# 面试练习记录",
            f"\n**日期：** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "\n---\n",
        ]
        for i, r in enumerate(self._rounds, 1):
            lines += [
                f"## 第 {i} 轮\n",
                f"### 面试官问题\n\n{r['question']}\n",
                f"### 我的回答\n\n{r['answer']}\n",
                f"### 评估师点评\n\n{r['evaluation']}\n",
                f"### 示范回答\n\n{r['better']}\n",
                "---\n",
            ]
        return "\n".join(lines)
