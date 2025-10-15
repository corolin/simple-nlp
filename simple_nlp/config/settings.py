"""
NLP搜索系统的配置管理。
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class SystemConfig(BaseSettings):
    """
    NLP搜索系统的配置设置。
    
    属性:
        data_file: JSONL数据文件的路径
        stopwords_file: 中文停用词文件的路径
        similarity_threshold: 搜索结果的最低相似度阈值
        top_k_results: 默认返回的结果数量
        max_query_length: 允许的最大查询长度
        cache_size: 用于存储结果的缓存大小
        preload_data: 是否在启动时预加载数据
        log_level: 日志级别
        host: 服务器主机地址
        port: 服务器端口
    """
    # File path configuration
    data_file: str = Field(default="data/data.jsonl", description="Path to the JSONL data file")
    stopwords_file: str = Field(default="data/stopwords", description="Path to the Chinese stopwords file or directory")
    
    # Algorithm parameters configuration
    similarity_threshold: float = Field(default=0.1, description="Minimum similarity threshold for search results")
    top_k_results: int = Field(default=5, description="Default number of results to return")
    max_query_length: int = Field(default=200, description="Maximum allowed query length")
    
    # Performance configuration
    cache_size: int = Field(default=1000, description="Size of the cache for storing results")
    preload_data: bool = Field(default=True, description="Whether to preload data on startup")
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    
    class Config:
        """Pydantic设置配置。"""
        env_file = ".env"
        env_prefix = "NLP_"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        """初始化配置，支持环境变量。"""
        super().__init__(**kwargs)
        
        # Ensure file paths are absolute
        self.data_file = os.path.abspath(self.data_file)
        self.stopwords_file = os.path.abspath(self.stopwords_file)
        
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """
        从环境变量创建配置。
        
        返回:
            SystemConfig: 配置实例
        """
        return cls()
    
    def validate_paths(self) -> bool:
        """
        验证配置的文件路径是否存在。
        
        返回:
            bool: 如果所有路径都有效则返回True，否则返回False
        """
        # 检查数据文件是否存在
        if not os.path.exists(self.data_file):
            print(f"警告：未找到数据文件 {self.data_file}")
            return False
            
        # 检查停用词文件是否存在
        if not os.path.exists(self.stopwords_file):
            print(f"警告：未找到停用词文件 {self.stopwords_file}")
            return False
            
        return True
    
    def __str__(self) -> str:
        """配置的字符串表示。"""
        return f"SystemConfig(data_file={self.data_file}, stopwords_file={self.stopwords_file}, " \
               f"similarity_threshold={self.similarity_threshold}, top_k_results={self.top_k_results})"
    
    def __repr__(self) -> str:
        """配置的表示。"""
        return self.__str__()