import os
import sys
import json
import re
import ast
import argparse
import logging
from typing import Any, Dict, List
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from openai import OpenAI

# Загрузка переменных из .env файла
load_dotenv()

# --- 1. Настройка логирования ---
logger = logging.getLogger(__name__)

def setup_logger() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

# --- 2. Настройка LLM (Локальный OpenAI-совместимый клиент Ollama) ---
def get_llm() -> RunnableLambda:
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("OPENAI_MODEL") or os.getenv("MODEL") or "qwen2.5-coder:7b"
    
    try:
        temperature = float(os.getenv("TEMPERATURE", "0.2"))
    except ValueError:
        temperature = 0.2

    client = OpenAI(api_key=api_key, base_url=base_url)
    logger.debug("Инициализация локального LLM: model=%s, base_url=%s", model, base_url)

    def _invoke(messages_or_value: Any) -> str:
        if hasattr(messages_or_value, "to_messages"):
            lc_messages = messages_or_value.to_messages()
        else:
            lc_messages = messages_or_value
            
        if not isinstance(lc_messages, list):
            raise TypeError("Ожидается список сообщений для LLM")
            
        role_map = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}
        oa_messages = []
        for m in lc_messages:
            role = "user"
            for cls, r in role_map.items():
                if isinstance(m, cls):
                    role = r
                    break
            content = m.content if isinstance(m.content, str) else str(m.content)
            oa_messages.append({"role": role, "content": content})
            
        resp = client.chat.completions.create(model=model, messages=oa_messages, temperature=temperature)
        return resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""

    return RunnableLambda(_invoke)

# --- 3. Цепочки (Chains) ---

def build_analysis_chain(llm):
    system = (
        "You are a senior data engineer and web scraping expert. Analyze the user's web scraping task. "
        "Output a concise JSON spec with keys: "
        "`target_url` (string), `data_to_extract` (list of strings), "
        "`anti_bot_protection` (boolean), `estimated_complexity` (low/medium/high), `notes`. "
        "Return ONLY valid minified JSON. No code fences."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Task: {task}")])
    return prompt | llm | StrOutputParser()

def build_tools_chain(llm):
    system = (
        "You are a Python solutions architect. Based on the scraping spec JSON, propose the best technical stack. "
        "Output compact JSON with keys: `main_library`, `additional_packages` (list), `proxy_needed` (boolean), `notes`. "
        "Return ONLY minified JSON. No code fences."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Spec JSON: {spec_json}")])
    return prompt | llm | StrOutputParser()

def build_implementation_chain(llm):
    system = (
        "You are a senior Python developer. Generate a complete, runnable web scraping script. "
        "STRICT REQUIREMENTS: 1. Use robust error handling (try/except). 2. Use logging module. "
        "3. Read sensitive data from environment variables using os.getenv. 4. No placeholders. "
        "Return ONLY valid Python code. No explanations. No code fences."
    )
    human = "Spec: {spec_json}\nTools: {tools_json}\nGenerate the full Python source code now."
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt | llm | StrOutputParser()

def build_review_chain(llm):
    system = (
        "You are a strict Python code reviewer. Check for: syntax errors, missing imports, "
        "lack of error handling, hardcoded secrets. Fix them and return the corrected full code. "
        "Return ONLY Python code. No explanations. No code fences."
    )
    human = "Issues found: {issues}\n\nOriginal code:\n{code}\n\nProvide corrected full code."
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt | llm | StrOutputParser()

# --- 4. Вспомогательные функции ---

def parse_json_spec(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Не удалось извлечь JSON из ответа LLM")

def strip_code_fences(text: str) -> str:
    pattern = r"^\s*```[a-zA-Z]*\s*\n|\n\s*```\s*$"
    return re.sub(pattern, "", text, flags=re.MULTILINE).strip()

def minimal_static_review(code: str) -> List[str]:
    issues: List[str] = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        issues.append(f"SyntaxError: {e}")
        return issues
    if "import" not in code:
        issues.append("Missing imports")
    if "try:" not in code and "except" not in code:
        issues.append("Missing error handling (try/except)")
    if "os.getenv" not in code and "os.environ" not in code:
        issues.append("Use os.getenv for sensitive data")
    if "logging" not in code:
        issues.append("Missing logging")
    return issues

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "scraper"

# --- 5. Запуск цепочки ---

def run_chain(task: str, out_dir: Path) -> Path:
    llm = get_llm()
    logger.info("🚀 Запуск цепочки генерации скрапера на локальном ИИ...")

    logger.info("🔍 Шаг 1: Анализ задачи...")
    analysis_text = build_analysis_chain(llm).invoke({"task": task})
    spec = parse_json_spec(analysis_text)
    spec_json = json.dumps(spec, ensure_ascii=False)
    logger.info(f"✅ Анализ завершен. Сложность: {spec.get('estimated_complexity')}")

    logger.info("🛠 Шаг 2: Подбор инструментов...")
    tools_text = build_tools_chain(llm).invoke({"spec_json": spec_json})
    tools = parse_json_spec(tools_text)
    tools_json = json.dumps(tools, ensure_ascii=False)
    logger.info(f"✅ Выбран инструмент: {tools.get('main_library')}")

    logger.info("💻 Шаг 3: Генерация кода...")
    code_text = build_implementation_chain(llm).invoke({"spec_json": spec_json, "tools_json": tools_json})
    code_text = strip_code_fences(code_text)

    logger.info("🔎 Шаг 4: Статическое ревью кода...")
    issues = minimal_static_review(code_text)
    if issues:
        logger.warning(f"⚠️ Найдены проблемы: {', '.join(issues)}. Отправка на исправление...")
        fixed_code = build_review_chain(llm).invoke({"issues": "\n".join(issues), "code": code_text})
        fixed_code = strip_code_fences(fixed_code)
        if len(minimal_static_review(fixed_code)) < len(issues):
            code_text = fixed_code
            logger.info("✅ Код успешно исправлен LLM-ревьюером.")

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"scraper_{slugify(task)[:30] or 'bot'}.py"
    out_path = out_dir / fname
    out_path.write_text(code_text, encoding="utf-8")
    logger.info(f"💾 Скрипт сохранен: {out_path.resolve()}")
    return out_path

def main(argv: List[str] | None = None) -> int:
    setup_logger()
    parser = argparse.ArgumentParser(description="Генерация Python-скрапера через LangChain (Local LLM).")
    parser.add_argument("task", type=str, help="Описание задачи скрапинга")
    parser.add_argument("--out-dir", type=str, default=".", help="Директория для сохранения")
    args = parser.parse_args(argv)

    try:
        run_chain(args.task, Path(args.out_dir))
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())