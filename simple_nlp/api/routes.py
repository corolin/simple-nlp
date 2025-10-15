"""
API routes for the NLP search system.
"""
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from simple_nlp.api.models import (
    SearchRequest, SearchResponse, SearchResult, SimilarityScore,
    QADataResponse, StopwordsRequest, StopwordsResponse,
    SystemStatus, ErrorResponse, HealthResponse, StatisticsResponse
)
from simple_nlp.core.search_engine import SearchEngine
from simple_nlp.config.settings import SystemConfig

logger = logging.getLogger(__name__)

# Global search engine instance
search_engine: Optional[SearchEngine] = None
start_time: float = time.time()


def get_search_engine() -> SearchEngine:
    """
    Get the global search engine instance.
    
    Returns:
        SearchEngine: Global search engine instance
        
    Raises:
        HTTPException: If the search engine is not initialized
    """
    global search_engine
    if search_engine is None:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    return search_engine


def create_router(config: SystemConfig) -> APIRouter:
    """
    Create API router with all endpoints.
    
    Args:
        config: System configuration
        
    Returns:
        APIRouter: Configured API router
    """
    router = APIRouter()
    
    @router.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """
        Health check endpoint.
        
        Returns:
            HealthResponse: Health status
        """
        try:
            engine = get_search_engine()
            status = "healthy" if engine.is_initialized else "unhealthy"
            
            return HealthResponse(
                status=status,
                message="System is running normally" if status == "healthy" else "System is not properly initialized",
                timestamp=datetime.now().isoformat()
            )
        except HTTPException:
            return HealthResponse(
                status="unhealthy",
                message="Search engine not initialized",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return HealthResponse(
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    @router.get("/status", response_model=SystemStatus, tags=["System"])
    async def get_status():
        """
        Get system status.
        
        Returns:
            SystemStatus: System status information
        """
        try:
            engine = get_search_engine()
            stats = engine.get_statistics()
            
            return SystemStatus(
                status="healthy" if engine.is_initialized else "unhealthy",
                data_loaded=engine.is_initialized,
                index_built=stats["similarity_index_built"],
                total_qa_pairs=stats["total_qa_pairs"],
                total_stopwords=stats["total_stopwords"],
                uptime=time.time() - start_time,
                version="1.0.0"
            )
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")
    
    @router.get("/statistics", response_model=StatisticsResponse, tags=["System"])
    async def get_statistics():
        """
        Get system statistics.
        
        Returns:
            StatisticsResponse: System statistics
        """
        try:
            engine = get_search_engine()
            stats = engine.get_statistics()
            
            return StatisticsResponse(
                total_qa_pairs=stats["total_qa_pairs"],
                total_stopwords=stats["total_stopwords"],
                total_custom_stopwords=stats["total_custom_stopwords"],
                similarity_index_built=stats["similarity_index_built"],
                similarity_threshold=stats["similarity_threshold"],
                default_top_k=stats["default_top_k"],
                max_query_length=stats["max_query_length"],
                uptime=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
    
    @router.post("/search", response_model=SearchResponse, tags=["Search"])
    async def search(request: SearchRequest):
        """
        Search for QA pairs similar to the query.
        
        Args:
            request: Search request containing query and parameters
            
        Returns:
            SearchResponse: Search results
        """
        try:
            engine = get_search_engine()
            
            # Perform search based on search type
            if request.search_type == "similarity":
                results = engine.search(request.query, request.top_k)
            elif request.search_type == "keyword":
                results = engine.search_by_keyword(request.query, request.top_k, request.field)
            elif request.search_type == "multi_keyword":
                # Split query into multiple keywords for multi-keyword search
                keywords = [kw.strip() for kw in request.query.split() if kw.strip()]
                results = engine.search_by_keywords(keywords, request.top_k, request.field, request.match_all)
            else:
                raise HTTPException(status_code=400, detail=f"Invalid search type: {request.search_type}")
            
            # Convert results to API format
            api_results = []
            for result in results:
                similarity_score = SimilarityScore(
                    tfidf_score=result.similarity_score.tfidf_score,
                    jaccard_score=result.similarity_score.jaccard_score,
                    bm25_score=result.similarity_score.bm25_score,
                    final_score=result.similarity_score.final_score,
                    match_keywords=result.similarity_score.match_keywords
                )
                
                api_result = SearchResult(
                    id=result.qa_data.id,
                    question=result.qa_data.question,
                    answer=result.qa_data.answer,
                    keywords=result.qa_data.keywords,
                    answer_keywords=result.qa_data.answer_keywords,
                    all_keywords=result.qa_data.all_keywords,
                    similarity=similarity_score,
                    rank=result.rank
                )
                
                api_results.append(api_result)
            
            return SearchResponse(
                results=api_results,
                total_matches=len(api_results),
                query=request.query,
                search_type=request.search_type
            )
            
        except ValueError as e:
            logger.warning(f"无效的搜索请求: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    @router.get("/qa/{qa_id}", response_model=QADataResponse, tags=["Data"])
    async def get_qa_by_id(qa_id: str):
        """
        Get QA data by ID.
        
        Args:
            qa_id: ID of the QA data to retrieve
            
        Returns:
            QADataResponse: QA data
        """
        try:
            engine = get_search_engine()
            qa_data = engine.get_qa_by_id(qa_id)
            
            if qa_data is None:
                raise HTTPException(status_code=404, detail=f"QA data with ID {qa_id} not found")
            
            return QADataResponse(
                id=qa_data.id,
                question=qa_data.question,
                answer=qa_data.answer,
                keywords=qa_data.keywords,
                answer_keywords=qa_data.answer_keywords,
                all_keywords=qa_data.all_keywords,
                create_time=qa_data.create_time.isoformat() if qa_data.create_time else ""
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取QA数据失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get QA data: {str(e)}")
    
    @router.get("/qa", response_model=List[QADataResponse], tags=["Data"])
    async def get_all_qa_data():
        """
        Get all QA data.
        
        Returns:
            List[QADataResponse]: List of all QA data
        """
        try:
            engine = get_search_engine()
            qa_data_list = engine.get_all_qa_data()
            
            return [
                QADataResponse(
                    id=qa_data.id,
                    question=qa_data.question,
                    answer=qa_data.answer,
                    keywords=qa_data.keywords,
                    answer_keywords=qa_data.answer_keywords,
                    all_keywords=qa_data.all_keywords,
                    create_time=qa_data.create_time.isoformat() if qa_data.create_time else ""
                )
                for qa_data in qa_data_list
            ]
            
        except Exception as e:
            logger.error(f"获取所有QA数据失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get all QA data: {str(e)}")
    
    @router.post("/stopwords", response_model=StopwordsResponse, tags=["Stopwords"])
    async def manage_stopwords(request: StopwordsRequest):
        """
        Add or remove custom stopwords.
        
        Args:
            request: Stopwords request containing words and action
            
        Returns:
            StopwordsResponse: Response with operation result
        """
        try:
            engine = get_search_engine()
            
            if request.action == "add":
                engine.add_custom_stopwords(request.words)
                message = f"Added {len(request.words)} custom stopwords"
            elif request.action == "remove":
                engine.remove_custom_stopwords(request.words)
                message = f"Removed {len(request.words)} custom stopwords"
            else:
                raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
            
            # Get updated statistics
            stats = engine.get_statistics()
            
            return StopwordsResponse(
                message=message,
                total_stopwords=stats["total_stopwords"],
                affected_words=request.words
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"管理停用词失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to manage stopwords: {str(e)}")
    
    @router.post("/reload", tags=["System"])
    async def reload_data():
        """
        Reload data and rebuild indexes.
        
        Returns:
            dict: Response with operation result
        """
        try:
            engine = get_search_engine()
            engine.reload_data()
            
            return {"message": "Data reloaded successfully"}
            
        except Exception as e:
            logger.error(f"重新加载数据失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to reload data: {str(e)}")
    
    async def global_exception_handler(request: Request, exc: Exception):
        """
        Global exception handler.
        
        Args:
            request: Request object
            exc: Exception object
            
        Returns:
            JSONResponse: Error response
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
    
    
    return router


def initialize_search_engine(config: SystemConfig) -> SearchEngine:
    """
    Initialize the global search engine instance.
    
    Args:
        config: System configuration
        
    Returns:
        SearchEngine: Initialized search engine
    """
    global search_engine
    
    if search_engine is None:
        search_engine = SearchEngine(config)
        search_engine.initialize()
        logger.info("搜索引擎初始化成功")
    
    return search_engine