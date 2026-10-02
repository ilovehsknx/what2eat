# src/recommendation/schemas.py
from typing import Annotated

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """推荐请求模型"""

    city: Annotated[str, Field(..., min_length=1, max_length=50, description="城市名称，例如: 北京")]
    model: Annotated[str | None, Field(None, description="Ollama 模型名称，为空则使用默认模型")]


class DishRecommendation(BaseModel):
    """单道推荐菜品"""

    name: Annotated[str, Field(description="菜品名称")]
    reason: Annotated[str, Field(description="推荐理由")]


class RecommendResponse(BaseModel):
    """推荐响应模型"""

    city: str
    weather: str
    temperature: str
    recommendations: list[DishRecommendation]
    model_used: str
