# 微型NLP搜索问答系统

基于自然语言处理的微型搜索问答系统，能够理解用户输入的对话内容，通过分词和语义分析检索JSONL数据源，返回高关联度的问答内容。

## 功能特性

- 🔍 **智能搜索**: 支持基于语义相似度的智能问答检索
- 🎯 **多种算法**: 集成TF-IDF、Jaccard相似度和BM25算法
- 📝 **中文处理**: 专业的中文分词和停用词处理
- 🌐 **多语言支持**: 支持中英文停用词和自定义停用词
- 🚀 **RESTful API**: 完整的REST API接口，支持JSON格式交互
- ⚙️ **配置管理**: 灵活的配置管理系统
- 📊 **实时统计**: 系统状态和统计信息监控
- 🛡️ **错误处理**: 完善的异常处理和错误响应机制
- 📦 **标准包结构**: 符合Python标准的包结构和模块化设计

## 系统架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  客户端请求      │ -> │   FastAPI服务层   │ -> │   NLP处理引擎    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   数据检索层     │ <- │   数据存储层     │
                       └─────────────────┘    └─────────────────┘
```

## 快速开始

### 环境要求

- Python 3.8+
- uv (推荐) 或 pip

### 安装依赖

```bash
# 使用uv安装依赖
uv sync --index-url https://repo.huaweicloud.com/repository/pypi/simple

# 或使用pip安装
pip install -e .
```

### 数据准备

1. 确保数据文件 `data/data.jsonl` 存在并包含问答数据
2. 确保停用词目录 `data/stopwords/` 存在并包含停用词文件

### 启动服务

```bash
# 使用CLI命令（推荐）
python -m simple_nlp.cli

# 或使用环境检查
python -m simple_nlp.cli --check

# 自定义端口和主机
python -m simple_nlp.cli --host 0.0.0.0 --port 8000

# 或直接启动主模块
python -m simple_nlp.main
```

### 访问API

服务启动后，可以通过以下地址访问：

- API文档: http://localhost:8000/docs
- ReDoc文档: http://localhost:8000/redoc
- 根路径: http://localhost:8000

## API接口

### 搜索问答

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是人工智能？",
    "top_k": 5,
    "search_type": "similarity"
  }'
```

### 系统状态

```bash
curl "http://localhost:8000/api/v1/status"
```

### 健康检查

```bash
curl "http://localhost:8000/api/v1/health"
```

## 配置说明

系统支持通过环境变量进行配置，所有配置项都以 `NLP_` 为前缀：

```bash
# 数据文件路径
export NLP_DATA_FILE="data/data.jsonl"

# 停用词文件或目录路径
export NLP_STOPWORDS_FILE="data/stopwords"

# 服务器配置
export NLP_HOST="0.0.0.0"
export NLP_PORT="8000"

# 算法参数
export NLP_SIMILARITY_THRESHOLD="0.1"
export NLP_TOP_K_RESULTS="5"
export NLP_MAX_QUERY_LENGTH="200"

# 日志级别
export NLP_LOG_LEVEL="INFO"
```

## 项目结构

```
simple-nlp/
├── data/                           # 数据文件目录
│   ├── data.jsonl                  # QA问答数据
│   └── stopwords/                   # 停用词目录
│       ├── cn_stopwords.txt         # 中文停用词
│       └── en_stopwords.txt         # 英文停用词
├── simple_nlp/                      # 主包目录
│   ├── __init__.py                  # 包初始化文件
│   ├── cli.py                       # 命令行接口
│   ├── main.py                      # FastAPI应用主入口
│   ├── api/                         # API接口模块
│   │   ├── __init__.py
│   │   ├── models.py                # API数据模型
│   │   └── routes.py                # API路由
│   ├── config/                      # 配置管理模块
│   │   ├── __init__.py
│   │   └── settings.py              # 系统配置
│   ├── core/                        # 核心功能模块
│   │   ├── __init__.py
│   │   ├── data_loader.py           # 数据加载器
│   │   ├── stopwords_manager.py     # 停用词管理器
│   │   ├── nlp_processor.py         # NLP处理器
│   │   ├── similarity_calculator.py # 相似度计算器
│   │   └── search_engine.py         # 搜索引擎
│   ├── models/                      # 数据模型模块
│   │   ├── __init__.py
│   │   └── qa_data.py               # QA数据模型
│   └── utils/                       # 工具模块
│       ├── __init__.py
│       ├── file_utils.py            # 文件处理工具
│       └── text_utils.py            # 文本处理工具
├── docs/                            # 文档目录
├── pyproject.toml                   # 项目配置文件
├── readme.md                        # 项目说明
├── LICENSE                          # 许可证
└── .gitignore                       # Git忽略文件
```

## 停用词说明

本系统支持多语言停用词过滤，停用词数据来源于开源项目：

### 数据来源
- **中文停用词**: [goto456/stopwords](https://github.com/goto456/stopwords)
- **英文停用词**: 基于标准英文停用词集合

### 停用词目录结构
```
data/stopwords/
├── cn_stopwords.txt    # 中文停用词（746个）
└── en_stopwords.txt    # 英文停用词（58个）
```

### 自定义停用词
系统支持以下方式添加自定义停用词：

1. **文件方式**: 在 `data/stopwords/` 目录下添加新的 `.txt` 文件
2. **API方式**: 通过API接口动态添加停用词

```python
# 通过API添加自定义停用词
curl -X POST "http://localhost:8000/api/v1/stopwords" \
  -H "Content-Type: application/json" \
  -d '{
    "words": ["自定义", "停用词"],
    "action": "add"
  }'
```

## 开发指南

### 安装开发环境

```bash
# 安装开发依赖
uv sync --dev

# 运行测试
pytest

# 代码格式化
black simple_nlp/
isort simple_nlp/

# 类型检查
mypy simple_nlp/
```

### 添加新的相似度算法

1. 在 `SimilarityCalculator` 类中添加新的计算方法
2. 更新 `SimilarityScore` 数据模型
3. 修改 `_calculate_final_score` 方法中的权重配置

### 扩展数据源

1. 实现 `DataLoader` 的子类
2. 重写 `_parse_qa_record` 方法以支持新的数据格式
3. 更新配置文件中的数据源路径

### 添加新的停用词

1. 在 `data/stopwords/` 目录下创建新的 `.txt` 文件
2. 每行一个停用词，支持 `#` 开头的注释行
3. 系统会自动加载目录下所有的 `.txt` 文件

## 性能优化

- 使用内存缓存提高响应速度
- 预计算TF-IDF矩阵减少实时计算开销
- 支持批量查询处理
- 可配置的相似度阈值平衡精度和性能
- 多语言停用词支持提高搜索精度

## 致谢

- [jieba](https://github.com/fxsjy/jieba) - 中文分词库
- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) - BM25算法实现
- [goto456/stopwords](https://github.com/goto456/stopwords) - 中文停用词数据