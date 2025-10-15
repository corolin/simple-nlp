"""
API models for the NLP search system.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class SearchRequest(BaseModel):
    """
    Search request model.
    
    Attributes:
        query: Search query text
        top_k: Number of top results to return
        search_type: Type of search ('similarity', 'keyword', or 'multi_keyword')
        field: Field to search in ('question', 'answer', or 'all')
        match_all: For multi-keyword search, whether to match all keywords
    """
    query: str = Field(..., description="Search query text", min_length=1, max_length=200)
    top_k: Optional[int] = Field(default=5, description="Number of top results to return", ge=1, le=50)
    search_type: Optional[str] = Field(default="similarity", description="Type of search ('similarity', 'keyword', or 'multi_keyword')")
    field: Optional[str] = Field(default="all", description="Field to search in ('question', 'answer', or 'all')")
    match_all: Optional[bool] = Field(default=False, description="For multi-keyword search, whether to match all keywords")
    
    @validator('search_type')
    def validate_search_type(cls, v):
        """验证search_type字段。"""
        if v not in ['similarity', 'keyword', 'multi_keyword']:
            raise ValueError("search_type must be one of: 'similarity', 'keyword', 'multi_keyword'")
        return v
    
    @validator('field')
    def validate_field(cls, v):
        """验证field字段。"""
        if v not in ['question', 'answer', 'all']:
            raise ValueError("field must be one of: 'question', 'answer', 'all'")
        return v


class SimilarityScore(BaseModel):
    """
    Similarity score model.
    
    Attributes:
        tfidf_score: TF-IDF similarity score
        jaccard_score: Jaccard similarity score
        bm25_score: BM25 similarity score
        final_score: Final combined similarity score
        match_keywords: List of matching keywords
    """
    tfidf_score: float = Field(..., description="TF-IDF similarity score", ge=0.0, le=1.0)
    jaccard_score: float = Field(..., description="Jaccard similarity score", ge=0.0, le=1.0)
    bm25_score: float = Field(..., description="BM25 similarity score", ge=0.0)
    final_score: float = Field(..., description="Final combined similarity score", ge=0.0, le=1.0)
    match_keywords: List[str] = Field(default_factory=list, description="List of matching keywords")


class SearchResult(BaseModel):
    """
    Search result model.
    
    Attributes:
        id: Unique identifier for the QA pair
        question: Question text
        answer: Answer text
        keywords: List of keywords extracted from the question
        similarity: Similarity score object
        rank: Rank of the result
    """
    id: str = Field(..., description="Unique identifier for the QA pair")
    question: str = Field(..., description="Question text")
    answer: str = Field(..., description="Answer text")
    keywords: List[str] = Field(default_factory=list, description="List of keywords extracted from the question")
    answer_keywords: List[str] = Field(default_factory=list, description="List of keywords extracted from the answer")
    all_keywords: List[str] = Field(default_factory=list, description="Combined list of keywords from both question and answer")
    similarity: SimilarityScore = Field(..., description="Similarity score object")
    rank: int = Field(..., description="Rank of the result", ge=1)


class SearchResponse(BaseModel):
    """
    Search response model.
    
    Attributes:
        results: List of search results
        total_matches: Total number of matches found
        query: Original query text
        search_type: Type of search performed
    """
    results: List[SearchResult] = Field(default_factory=list, description="List of search results")
    total_matches: int = Field(..., description="Total number of matches found", ge=0)
    query: str = Field(..., description="Original query text")
    search_type: str = Field(..., description="Type of search performed")


class QADataResponse(BaseModel):
    """
    QA data response model.
    
    Attributes:
        id: Unique identifier for the QA pair
        question: Question text
        answer: Answer text
        keywords: List of keywords extracted from the question
        create_time: Creation timestamp
    """
    id: str = Field(..., description="Unique identifier for the QA pair")
    question: str = Field(..., description="Question text")
    answer: str = Field(..., description="Answer text")
    keywords: List[str] = Field(default_factory=list, description="List of keywords extracted from the question")
    answer_keywords: List[str] = Field(default_factory=list, description="List of keywords extracted from the answer")
    all_keywords: List[str] = Field(default_factory=list, description="Combined list of keywords from both question and answer")
    create_time: str = Field(..., description="Creation timestamp")


class StopwordsRequest(BaseModel):
    """
    Stopwords request model.
    
    Attributes:
        words: List of stopwords to add or remove
        action: Action to perform ('add' or 'remove')
    """
    words: List[str] = Field(..., description="List of stopwords to add or remove", min_items=1)
    action: str = Field(..., description="Action to perform ('add' or 'remove')")
    
    @validator('action')
    def validate_action(cls, v):
        """验证action字段。"""
        if v not in ['add', 'remove']:
            raise ValueError("action must be either 'add' or 'remove'")
        return v


class StopwordsResponse(BaseModel):
    """
    Stopwords response model.
    
    Attributes:
        message: Response message
        total_stopwords: Total number of stopwords after the operation
        affected_words: List of affected words
    """
    message: str = Field(..., description="Response message")
    total_stopwords: int = Field(..., description="Total number of stopwords after the operation", ge=0)
    affected_words: List[str] = Field(default_factory=list, description="List of affected words")


class SystemStatus(BaseModel):
    """
    System status model.
    
    Attributes:
        status: System status ('healthy' or 'unhealthy')
        data_loaded: Whether data is loaded
        index_built: Whether similarity index is built
        total_qa_pairs: Total number of QA pairs
        total_stopwords: Total number of stopwords
        uptime: System uptime in seconds
        version: System version
    """
    status: str = Field(..., description="System status ('healthy' or 'unhealthy')")
    data_loaded: bool = Field(..., description="Whether data is loaded")
    index_built: bool = Field(..., description="Whether similarity index is built")
    total_qa_pairs: int = Field(..., description="Total number of QA pairs", ge=0)
    total_stopwords: int = Field(..., description="Total number of stopwords", ge=0)
    uptime: float = Field(..., description="System uptime in seconds", ge=0.0)
    version: str = Field(..., description="System version")


class ErrorResponse(BaseModel):
    """
    Error response model.
    
    Attributes:
        error: Error message
        error_code: Error code
        details: Additional error details
    """
    error: str = Field(..., description="Error message")
    error_code: int = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class HealthResponse(BaseModel):
    """
    Health check response model.
    
    Attributes:
        status: System status ('healthy' or 'unhealthy')
        message: Status message
        timestamp: Current timestamp
    """
    status: str = Field(..., description="System status ('healthy' or 'unhealthy')")
    message: str = Field(..., description="Status message")
    timestamp: str = Field(..., description="Current timestamp")


class StatisticsResponse(BaseModel):
    """
    Statistics response model.
    
    Attributes:
        total_qa_pairs: Total number of QA pairs
        total_stopwords: Total number of stopwords
        total_custom_stopwords: Total number of custom stopwords
        similarity_index_built: Whether similarity index is built
        similarity_threshold: Similarity threshold for search results
        default_top_k: Default number of top results to return
        max_query_length: Maximum allowed query length
        uptime: System uptime in seconds
    """
    total_qa_pairs: int = Field(..., description="Total number of QA pairs", ge=0)
    total_stopwords: int = Field(..., description="Total number of stopwords", ge=0)
    total_custom_stopwords: int = Field(..., description="Total number of custom stopwords", ge=0)
    similarity_index_built: bool = Field(..., description="Whether similarity index is built")
    similarity_threshold: float = Field(..., description="Similarity threshold for search results", ge=0.0, le=1.0)
    default_top_k: int = Field(..., description="Default number of top results to return", ge=1)
    max_query_length: int = Field(..., description="Maximum allowed query length", ge=1)
    uptime: float = Field(..., description="System uptime in seconds", ge=0.0)