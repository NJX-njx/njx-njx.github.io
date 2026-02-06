# -*- coding: utf-8 -*-
"""
Multi-provider LLM helper for Njx'Log blog assistant.

Supported providers:
  - doubao   (Volcengine Ark — Doubao-Seed-1.8)
  - deepseek (DeepSeek official)
  - openai   (OpenAI)

Usage:
    # Set the API key for your provider as an environment variable:
    #   ARK_API_KEY      — for doubao
    #   DEEPSEEK_API_KEY — for deepseek
    #   OPENAI_API_KEY   — for openai
    pip install openai
    python llm_model.py                      # interactive mode (default: doubao)
    python llm_model.py --provider deepseek  # switch provider
"""

import os
import argparse
from openai import OpenAI

# ============ Provider configs ============
PROVIDERS = {
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-1-8-251228",
        "env_key": "ARK_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
}

DEFAULT_PROVIDER = "doubao"

# ============ System prompt ============
SYSTEM_PROMPT = """你是 Njx'Log 博客的 AI 助手，由倪家兴（Jiaxing Ni）创建。

## 你的身份
- 你是一个专注于技术博客的智能助手
- 博客主题涵盖：大模型（LLM）、多模态AI、RAG系统、机器学习等

## 你的职责
1. 根据提供的博客文章内容回答用户问题
2. 帮助用户理解技术概念
3. 推荐相关的博客文章
4. 如果问题超出博客内容范围，提供通用但有帮助的回答

## 回答风格
- 使用清晰、简洁的语言
- 适当使用 Markdown 格式（加粗、列表、代码块等）
- 引用文章时注明来源
- 如果不确定，诚实说明
- 支持中英文双语回答，根据用户语言自动切换

## 博客作者背景
- 北京理工大学 计算机/徐特立英才班
- 专注于：LLM 微调、RAG、多模态融合、RLHF
- 有百度 Hackathon、字节 Xpert 等项目经验"""


def create_client(provider: str = DEFAULT_PROVIDER):
    """Create an OpenAI-compatible client for the given provider."""
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["env_key"])
    if not api_key:
        raise EnvironmentError(
            f"Environment variable {cfg['env_key']} is not set. "
            f"Please export it before running."
        )
    return OpenAI(api_key=api_key, base_url=cfg["base_url"]), cfg["model"]


def chat(
    user_message: str,
    context: str | None = None,
    stream: bool = True,
    provider: str = DEFAULT_PROVIDER,
):
    """Send a message and optionally stream the response."""
    client, model = create_client(provider)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        user_content = (
            f"以下是相关的博客文章内容：\n\n{context}\n\n---\n\n用户问题：{user_message}"
        )
    else:
        user_content = user_message

    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
        max_tokens=20000,
        temperature=0.7,
    )

    if stream:
        full_response = ""
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                print(delta.reasoning_content, end="", flush=True)
            elif delta.content:
                print(delta.content, end="", flush=True)
                full_response += delta.content
        print()
        return full_response
    else:
        return response.choices[0].message.content


def chat_with_blog_context(
    user_message: str,
    articles: list,
    provider: str = DEFAULT_PROVIDER,
):
    """RAG-style chat with blog article context."""
    context_parts = []
    for i, article in enumerate(articles, 1):
        context_parts.append(
            f"【文章 {i}】{article.get('title', 'Untitled')}\n"
            f"链接：{article.get('url', '')}\n"
            f"内容：{article.get('content', '')[:1500]}..."
        )
    context = "\n\n---\n\n".join(context_parts) if context_parts else None
    return chat(user_message, context=context, stream=True, provider=provider)


def interactive_chat(provider: str = DEFAULT_PROVIDER):
    """Interactive REPL chat."""
    print("=" * 50)
    print(f"🤖 Njx'Log 博客 AI 助手  (provider: {provider})")
    print("=" * 50)
    print("输入问题开始对话，输入 'quit' 或 'exit' 退出\n")

    while True:
        try:
            user_input = input("📝 你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break
            print("\n🤖 AI: ", end="")
            chat(user_input, provider=provider)
            print()
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Njx'Log blog AI assistant")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default=DEFAULT_PROVIDER,
        help=f"LLM provider (default: {DEFAULT_PROVIDER})",
    )
    args = parser.parse_args()
    interactive_chat(provider=args.provider)
