#!/usr/bin/env python3
"""
命令行接口模块 - 微型NLP搜索问答系统
"""
import os
import sys
import argparse
import logging

from simple_nlp.config.settings import SystemConfig
from simple_nlp.main import create_app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """检查运行环境"""
    logger.info("检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        logger.error(f"Python版本过低: {sys.version}. 需要Python 3.8+")
        return False
    
    logger.info(f"Python版本: {sys.version}")
    
    # 检查必要文件
    config = SystemConfig.from_env()
    
    if not os.path.exists(config.data_file):
        logger.error(f"数据文件不存在: {config.data_file}")
        return False
    
    if not os.path.exists(config.stopwords_file):
        logger.error(f"停用词文件不存在: {config.stopwords_file}")
        return False
    
    logger.info("环境检查通过")
    return True


def run_server(host=None, port=None, log_level=None):
    """运行服务器"""
    # 加载配置
    config = SystemConfig.from_env()
    
    # 覆盖配置参数
    if host:
        config.host = host
    if port:
        config.port = port
    if log_level:
        config.log_level = log_level
    
    # 验证文件路径
    if not config.validate_paths():
        logger.error("配置文件路径验证失败")
        return False
    
    # 创建并运行应用
    try:
        import uvicorn
        app = create_app(config)
        
        logger.info(f"启动服务器: http://{config.host}:{config.port}")
        logger.info(f"API文档: http://{config.host}:{config.port}/docs")
        
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level=config.log_level.lower()
        )
        
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"启动服务器失败: {e}")
        return False
    
    return True


def main():
    """主命令行入口"""
    parser = argparse.ArgumentParser(description="微型NLP搜索问答系统")
    
    parser.add_argument(
        "--host",
        default=None,
        help="服务器主机地址 (默认: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="服务器端口 (默认: 8000)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别 (默认: INFO)"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅检查环境，不启动服务器"
    )
    
    args = parser.parse_args()
    
    # 显示欢迎信息
    print("=" * 60)
    print("微型NLP搜索问答系统")
    print("=" * 60)
    
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    if args.check:
        logger.info("环境检查完成，退出")
        return
    
    # 启动服务器
    if not run_server(args.host, args.port, args.log_level):
        sys.exit(1)


if __name__ == "__main__":
    main()