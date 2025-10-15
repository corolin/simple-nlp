"""
Simple NLP搜索系统的主入口点。
"""
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from simple_nlp.config.settings import SystemConfig
from simple_nlp.api.routes import create_router, initialize_search_engine
from simple_nlp.api.models import ErrorResponse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config: SystemConfig) -> FastAPI:
    """
    创建并配置FastAPI应用程序。
    
    参数:
        config: 系统配置
        
    返回:
        FastAPI: 配置好的FastAPI应用程序
    """
    # 创建FastAPI应用
    app = FastAPI(
        title="微型NLP搜索问答系统",
        description="基于自然语言处理的微型搜索问答系统，能够理解用户输入的对话内容，通过分词和语义分析检索JSONL数据源，返回高关联度的问答内容。",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 创建并包含API路由器
    router = create_router(config)
    app.include_router(router, prefix="/api/v1")
    
    @app.on_event("startup")
    async def startup_event():
        """
        启动时初始化搜索引擎。
        """
        logger.info("启动NLP搜索系统")
        
        try:
            # 初始化搜索引擎
            initialize_search_engine(config)
            logger.info("搜索引擎初始化成功")
        except Exception as e:
            logger.error(f"搜索引擎初始化失败: {e}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """
        关闭时清理资源。
        """
        logger.info("关闭NLP搜索系统")
    
    @app.get("/")
    async def root():
        """
        根端点。
        
        返回:
            dict: 欢迎消息
        """
        return {
            "message": "欢迎使用微型NLP搜索问答系统",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    
    # 添加全局异常处理器
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        全局异常处理器。
        
        参数:
            request: 请求对象
            exc: 异常对象
            
        返回:
            JSONResponse: 错误响应
        """
        logger.error(f"未处理的异常: {exc}")
        
        error_response = ErrorResponse(
            error="Internal server error",
            error_code=500,
            details={"exception": str(exc)}
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )
    
    return app


def main():
    """
    应用程序的主入口点。
    """
    # 加载配置
    config = SystemConfig.from_env()
    
    # 验证文件路径
    if not config.validate_paths():
        logger.error("配置中的文件路径无效")
        return
    
    # 创建FastAPI应用
    app = create_app(config)
    
    # 运行服务器
    logger.info(f"启动服务器: {config.host}:{config.port}")
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower()
    )


if __name__ == "__main__":
    main()