"""
Simple NLP搜索问答系统

一个基于自然语言处理的微型搜索问答系统，能够理解用户输入的对话内容，
通过分词和语义分析检索JSONL数据源，返回高关联度的问答内容。
"""

__version__ = "0.1.0"
__author__ = "Developer"
__email__ = "developer@example.com"

from simple_nlp.core.search_engine import SearchEngine
from simple_nlp.config.settings import SystemConfig

__all__ = [
    "SearchEngine",
    "SystemConfig",
    "__version__",
]