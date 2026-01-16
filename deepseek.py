# -*- coding: utf-8 -*-
"""
DeepSeek V3 API 调用示例 - Njx'Log 博客智能助手
通过百度 AI Studio 调用 DeepSeek 模型

Usage:
    pip install openai
    python deepseek.py
"""

from openai import OpenAI

# ============ API 配置 ============
API_KEY = "your-api-key-here"  # 请替换为你的 Access Token
BASE_URL = "https://aistudio.baidu.com/llm/lmapi/v3"
MODEL = "deepseek-v3"

# ============ 系统提示词 ============
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


def create_client():
    """创建 OpenAI 兼容客户端"""
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )


def chat(user_message: str, context: str = None, stream: bool = True):
    """
    与 DeepSeek 模型对话
    
    Args:
        user_message: 用户输入的问题
        context: 博客文章上下文（可选）
        stream: 是否使用流式输出
    
    Returns:
        如果 stream=False，返回完整回复；否则逐字打印并返回完整内容
    """
    client = create_client()
    
    # 构建消息
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # 如果有上下文，添加到用户消息中
    if context:
        user_content = f"""以下是相关的博客文章内容：

{context}

---

用户问题：{user_message}"""
    else:
        user_content = user_message
    
    messages.append({"role": "user", "content": user_content})
    
    # 调用 API
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=stream,
        extra_body={
            "web_search": {"enable": False}  # 博客问答场景关闭网络搜索
        },
        max_completion_tokens=2000,
        temperature=0.7
    )
    
    if stream:
        # 流式输出
        full_response = ""
        for chunk in response:
            if not chunk.choices or len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            # 处理推理内容（如果有）
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                print(delta.reasoning_content, end="", flush=True)
            # 处理正常内容
            elif delta.content:
                print(delta.content, end="", flush=True)
                full_response += delta.content
        print()  # 换行
        return full_response
    else:
        # 非流式输出
        return response.choices[0].message.content


def chat_with_blog_context(user_message: str, articles: list):
    """
    带博客文章上下文的对话（RAG 模式）
    
    Args:
        user_message: 用户问题
        articles: 相关文章列表，每个元素为 dict，包含 title, content, url
    
    Returns:
        模型回复内容
    """
    # 构建上下文
    context_parts = []
    for i, article in enumerate(articles, 1):
        context_parts.append(
            f"【文章 {i}】{article.get('title', 'Untitled')}\n"
            f"链接：{article.get('url', '')}\n"
            f"内容：{article.get('content', '')[:1500]}..."
        )
    
    context = "\n\n---\n\n".join(context_parts) if context_parts else None
    
    return chat(user_message, context=context, stream=True)


def interactive_chat():
    """交互式对话模式"""
    print("=" * 50)
    print("🤖 Njx'Log 博客 AI 助手")
    print("=" * 50)
    print("输入问题开始对话，输入 'quit' 或 'exit' 退出\n")
    
    while True:
        try:
            user_input = input("📝 你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            print("\n🤖 AI: ", end="")
            chat(user_input)
            print()
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break


# ============ 测试示例 ============
if __name__ == "__main__":
    # 运行交互式对话
    interactive_chat()
    
    # 或者测试单次问答：
    # print("📝 测试问题: 介绍一下这个博客")
    # print("-" * 30)
    # chat("介绍一下这个博客和博主")
