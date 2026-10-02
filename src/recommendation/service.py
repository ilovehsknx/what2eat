# src/recommendation/service.py
import json

import httpx
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.collections.model import Collection
from src.core.config import settings
from src.dishes.model import Dish
from src.recommendation.schemas import DishRecommendation, RecommendResponse
from src.weather.service import fetch_weather


async def get_user_favorites(session: AsyncSession, user_id) -> list[Dish]:
    """
    获取用户所有收藏夹中的菜品（去重）。

    遍历用户的所有收藏夹，收集其中的菜品，
    由于同一道菜可能被放在多个收藏夹中，需要按 id 去重。
    """
    # 查询用户的所有收藏夹，同时预加载关联的菜品（避免 N+1 查询）
    query = (
        select(Collection)
        .where(Collection.user_id == user_id)
    )
    result = await session.scalars(query)
    collections = result.all()

    # 用字典去重，key 为菜品 id
    dishes_map: dict[int, Dish] = {}
    for collection in collections:
        for dish in collection.dishes:
            dishes_map[dish.id] = dish

    return list(dishes_map.values())


def build_prompt(dishes: list[Dish], weather_info: dict) -> str:
    """
    构造发送给大模型的提示词。

    将用户的收藏菜品列表和当日天气信息整合成一段结构化的 prompt，
    引导大模型从收藏中推荐适合当前天气的菜品。
    """
    # 拼接菜品列表
    dish_list = "\n".join(
        f"{i + 1}. {dish.name} - {dish.description or '暂无描述'}"
        for i, dish in enumerate(dishes)
    )

    # 拼接天气信息
    weather_block = (
        f"城市：{weather_info['城市']}\n"
        f"天气：{weather_info['天气']}\n"
        f"气温：{weather_info['最低气温']} ~ {weather_info['最高气温']}"
    )

    return f"""你是一个专业的美食推荐助手。请根据以下信息，从用户的收藏菜品中推荐今天最适合吃的菜。

## 今日天气
{weather_block}

## 用户收藏的菜品
{dish_list}

## 推荐要求
1. 从上面的收藏列表中选择 2-3 道最适合今天天气的菜品
2. 根据气温、天气状况（如雨天适合暖身菜、高温天适合清爽菜等）给出推荐理由
3. 推荐理由要简洁有趣，控制在一句话以内

请严格按以下 JSON 格式返回，不要包含其他内容：
{{
  "recommendations": [
    {{"name": "菜品名", "reason": "推荐理由"}}
  ]
}}"""


async def call_ollama(prompt: str, model: str | None = None) -> str:
    """
    调用本地 Ollama API 生成推荐。

    通过 httpx 异步客户端向本地 Ollama 服务发送请求，
    使用 /api/chat 接口进行对话式生成。
    未来可轻松替换为其他 AI API（如 OpenAI、Claude 等）。
    """
    model_name = model or settings.ollama_model
    url = f"{settings.ollama_base_url}/api/chat"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,  # 非流式返回，一次性获取完整结果
    }

    logger.info(f"调用 Ollama API: model={model_name}")

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()

    data = response.json()
    content = data.get("message", {}).get("content", "")

    logger.debug(f"Ollama 返回内容: {content[:200]}...")
    return content


def parse_recommendations(raw_text: str) -> list[DishRecommendation]:
    """
    解析大模型返回的 JSON 文本。

    大模型返回的内容可能包含 markdown 代码块标记（```json ... ```），
    需要先清理再解析为 Python 对象。
    """
    # 清理 markdown 代码块标记
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # 去掉首行的 ```json 和末尾的 ```
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
        items = data.get("recommendations", [])
        return [DishRecommendation(**item) for item in items]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"解析 Ollama 返回内容失败: {e}")
        logger.debug(f"原始返回内容: {raw_text}")
        return []


async def get_recommendation(
    session: AsyncSession,
    user_id,
    city: str,
    http_client: httpx.AsyncClient,
    model: str | None = None,
) -> RecommendResponse:
    """
    核心推荐流程：获取收藏 → 获取天气 → 构造 prompt → 调用大模型 → 解析结果。

    这是整个推荐功能的入口函数，串联了所有步骤。
    """
    # 1. 获取用户收藏的菜品
    dishes = await get_user_favorites(session, user_id)
    if not dishes:
        logger.warning(f"用户 {user_id} 没有收藏任何菜品")
        # 返回空推荐，让调用方决定如何处理
        raise ValueError("用户没有收藏任何菜品，请先添加收藏")

    # 2. 获取指定城市的天气
    # 复用 weather 模块的 fetch_weather 函数，避免重复代码
    weather_info = await fetch_weather(http_client, city)

    if weather_info is None:
        raise ValueError(f"无法获取 {city} 的天气信息，请检查城市名称是否正确")

    # 3. 构造 prompt
    prompt = build_prompt(dishes, weather_info)

    # 4. 调用大模型
    raw_result = await call_ollama(prompt, model)

    # 5. 解析返回结果
    recommendations = parse_recommendations(raw_result)

    # 组装最终响应
    temperature = f"{weather_info['最低气温']} ~ {weather_info['最高气温']}"
    return RecommendResponse(
        city=weather_info["城市"],
        weather=weather_info["天气"],
        temperature=temperature,
        recommendations=recommendations,
        model_used=model or settings.ollama_model,
    )
