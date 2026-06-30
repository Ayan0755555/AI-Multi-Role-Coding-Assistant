import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

load_dotenv()


class LangChainClient:
    def __init__(self, mode="General"):

        role_prompts = {
            "Code Analysis": "You are an expert code analyst. Break down and explain code clearly.",
            "Code Generator": "You are a senior software engineer. Generate clean, production-ready code with comments.",
            "Debugger": "You are a debugging expert. Find bugs, explain them, and provide fixes.",
            "Code Guide": "You are a coding mentor. Teach programming step by step with simple explanations.",
            "Optimization": "You are a performance engineer. Optimize code for speed, memory, and readability.",
            "Explain Code": "You are a programming teacher. Explain every line of code with examples.",
            "Project Builder": "You are a full-stack developer. Build complete projects with proper file structure.",
            "Documentation": "You are a technical writer. Generate professional documentation and README files.",
            "General": "You are a helpful AI coding assistant."
        }

        self.system_prompt = role_prompts.get(mode, role_prompts["General"])

        self.model = ChatOpenAI(
            model="cohere/north-mini-code:free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
            max_tokens=2048,
        )

    def chat(self, messages):
        prompt_messages = [SystemMessage(content=self.system_prompt)]

        for msg in messages:
            if msg["role"] == "user":
                prompt_messages.append(
                    HumanMessage(content=msg["content"])
                )
            else:
                prompt_messages.append(
                    AIMessage(content=msg["content"])
                )

        try:
            response = self.model.invoke(prompt_messages)
            return response.content

        except Exception as e:
            return f"❌ Error: {str(e)}"