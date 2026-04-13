"""诊断脚本：测试API连接是否正常"""
import os

# 从 .env 文件加载配置（如果环境变量未设置）
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

import anthropic

base_url = os.environ.get("ANTHROPIC_BASE_URL")
api_key  = os.environ.get("ANTHROPIC_API_KEY")
model    = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

print(f"BASE_URL : {base_url}")
print(f"API_KEY  : {api_key[:10]}..." if api_key else "API_KEY  : 未设置")
print(f"MODEL    : {model}")
print()

client = anthropic.Anthropic()

print(">>> 测试 API 调用...")
try:
    resp = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[{"role": "user", "content": "说一个字"}],
    )
    print("成功！回复：", resp.content[0].text)
except Exception as e:
    print(f"失败：{type(e).__name__}")
    print(str(e))
