# src/recommendation/router.py
from typing import Annotated, cast

import httpx
from fastapi import APIRouter, Depends, Request
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.schemas import UserRead
from src.auth.user_manager import get_current_user
from src.core.database import get_db
from src.core.exception import BadRequestException
from src.recommendation.schemas import RecommendRequest, RecommendResponse
from src.recommendation.service import get_recommendation

router = APIRouter(prefix="/recommend", tags=["Recommendation"])


@router.post("", response_model=RecommendResponse, summary="根据天气推荐菜品")
async def recommend_dish(
    request_data: RecommendRequest,
    request: Request,
    current_user: Annotated[UserRead, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> RecommendResponse:
    """
    根据用户收藏和当日天气，调用大模型智能推荐菜品。

    - 从数据库读取用户所有收藏夹中的菜品
    - 调用天气 API 获取指定城市的天气
    - 将菜品列表和天气信息发送给本地 Ollama 大模型
    - 解析返回结果，推荐 2-3 道最适合的菜品
    """
    try:
        # 从 request.state 获取 lifespan 中创建的 httpx 客户端
        http_client = cast(httpx.AsyncClient, request.state.http_client)

        result = await get_recommendation(
            session=session,
            user_id=current_user.id,
            city=request_data.city,
            model=request_data.model,
            http_client=http_client,
        )
        return result

    except ValueError as e:
        # 业务逻辑错误（如没有收藏、天气获取失败）
        raise BadRequestException(detail=str(e))
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama API 调用失败: {e.response.status_code}")
        raise BadRequestException(detail="大模型服务调用失败，请稍后再试")
    except httpx.ConnectError:
        logger.error("无法连接到 Ollama 服务")
        raise BadRequestException(detail="无法连接到大模型服务，请确认 Ollama 已启动")
