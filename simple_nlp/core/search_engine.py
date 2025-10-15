"""
Search engine module for the NLP search system.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

from simple_nlp.models.qa_data import QAData
from simple_nlp.core.data_loader import DataLoader
from simple_nlp.core.stopwords_manager import StopWordsManager
from simple_nlp.core.nlp_processor import NLPProcessor
from simple_nlp.core.similarity_calculator import SimilarityCalculator, SimilarityScore
from simple_nlp.config.settings import SystemConfig

logger = logging.getLogger(__name__)


class SearchResult:
    """
    Data class to store search results.
    
    Attributes:
        qa_data: QA data object
        similarity_score: Similarity score object
        rank: Rank of the result
    """
    
    def __init__(self, qa_data: QAData, similarity_score: SimilarityScore, rank: int):
        self.qa_data = qa_data
        self.similarity_score = similarity_score
        self.rank = rank
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert search result to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the search result
        """
        return {
            "id": self.qa_data.id,
            "question": self.qa_data.question,
            "answer": self.qa_data.answer,
            "keywords": self.qa_data.keywords,
            "similarity": {
                "tfidf_score": self.similarity_score.tfidf_score,
                "jaccard_score": self.similarity_score.jaccard_score,
                "bm25_score": self.similarity_score.bm25_score,
                "final_score": self.similarity_score.final_score,
                "match_keywords": self.similarity_score.match_keywords
            },
            "rank": self.rank
        }


class SearchEngine:
    """
    Main search engine for the NLP search system.
    
    This class is responsible for:
    - Query parsing and processing
    - Multi-strategy retrieval execution
    - Result sorting and filtering
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the SearchEngine with system configuration.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        # Initialize components
        self.data_loader = DataLoader(config.data_file)
        self.stopwords_manager = StopWordsManager(config.stopwords_file)
        self.nlp_processor = NLPProcessor(self.stopwords_manager)
        self.similarity_calculator = SimilarityCalculator(self.nlp_processor)
        
        # Search state
        self.is_initialized = False
        self.qa_data_list: List[QAData] = []
        
    def initialize(self) -> None:
        """
        Initialize the search engine by loading data and building indexes.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info("初始化搜索引擎")
            
            # Load stopwords
            self.stopwords_manager.load_stopwords()
            
            # Load QA data
            self.data_loader.load_data()
            self.qa_data_list = self.data_loader.get_all_qa_data()
            
            # Build similarity index for combined text (question + answer)
            combined_texts = []
            for qa in self.qa_data_list:
                # Combine question and answer for better search coverage
                combined_text = f"{qa.question} {qa.answer}"
                combined_texts.append(combined_text)
            
            self.similarity_calculator.build_index(combined_texts)
            
            self.is_initialized = True
            logger.info(f"搜索引擎初始化完成，加载了 {len(self.qa_data_list)} 条QA记录")
            
        except Exception as e:
            logger.error(f"搜索引擎初始化失败: {e}")
            raise
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Search for QA pairs similar to the query.
        
        Args:
            query: Search query text
            top_k: Number of top results to return (uses config default if None)
            
        Returns:
            List[SearchResult]: List of search results
            
        Raises:
            RuntimeError: If the search engine has not been initialized
            ValueError: If the query is invalid
        """
        if not self.is_initialized:
            raise RuntimeError("Search engine has not been initialized")
            
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        # Use default top_k if not provided
        if top_k is None:
            top_k = self.config.top_k_results
            
        # Validate query length
        if len(query) > self.config.max_query_length:
            raise ValueError(f"Query exceeds maximum length of {self.config.max_query_length}")
        
        try:
            logger.info(f"正在搜索: {query}")
            
            # Search for similar documents
            similar_docs = self.similarity_calculator.search_similar_documents(query, top_k)
            
            # Convert to search results
            results = []
            for rank, (doc_index, similarity_score) in enumerate(similar_docs, 1):
                # Get QA data
                qa_data = self.qa_data_list[doc_index]
                
                # Apply similarity threshold
                if similarity_score.final_score >= self.config.similarity_threshold:
                    result = SearchResult(qa_data, similarity_score, rank)
                    results.append(result)
            
            logger.info(f"找到 {len(results)} 条匹配查询 '{query}' 的结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索过程中出错: {e}")
            raise
    
    def search_by_keyword(self, keyword: str, top_k: Optional[int] = None, field: str = 'all') -> List[SearchResult]:
        """
        Search for QA pairs containing a specific keyword.
        
        Args:
            keyword: Keyword to search for
            top_k: Number of top results to return (uses config default if None)
            field: Field to search in ('question', 'answer', or 'all')
            
        Returns:
            List[SearchResult]: List of search results
            
        Raises:
            RuntimeError: If the search engine has not been initialized
            ValueError: If the keyword is invalid
        """
        if not self.is_initialized:
            raise RuntimeError("Search engine has not been initialized")
            
        if not keyword or not keyword.strip():
            raise ValueError("Keyword cannot be empty")
            
        # Use default top_k if not provided
        if top_k is None:
            top_k = self.config.top_k_results
        
        try:
            logger.info(f"在字段 '{field}' 中搜索关键词 '{keyword}'")
            
            # Search for QA data containing the keyword
            matching_qa_data = self.data_loader.search_by_keyword(keyword, field)
            
            # Convert to search results
            results = []
            for rank, qa_data in enumerate(matching_qa_data[:top_k], 1):
                # Create a similarity score based on field match
                field_boost = 1.0
                if field == 'question' and qa_data.has_keyword(keyword, 'question'):
                    field_boost = 1.2  # Boost for exact question match
                elif field == 'answer' and qa_data.has_keyword(keyword, 'answer'):
                    field_boost = 1.2  # Boost for exact answer match
                
                similarity_score = SimilarityScore(
                    tfidf_score=field_boost,
                    jaccard_score=field_boost,
                    bm25_score=field_boost,
                    final_score=field_boost,
                    match_keywords=[keyword]
                )
                
                result = SearchResult(qa_data, similarity_score, rank)
                results.append(result)
            
            logger.info(f"在字段 '{field}' 中找到 {len(results)} 条匹配关键词 '{keyword}' 的结果")
            return results
            
        except Exception as e:
            logger.error(f"关键词搜索过程中出错: {e}")
            raise
    
    def search_by_keywords(self, keywords: List[str], top_k: Optional[int] = None,
                          field: str = 'all', match_all: bool = False) -> List[SearchResult]:
        """
        Search for QA pairs containing multiple keywords.
        
        Args:
            keywords: List of keywords to search for
            top_k: Number of top results to return (uses config default if None)
            field: Field to search in ('question', 'answer', or 'all')
            match_all: If True, require all keywords to match; if False, any keyword match is sufficient
            
        Returns:
            List[SearchResult]: List of search results
            
        Raises:
            RuntimeError: If the search engine has not been initialized
            ValueError: If keywords list is empty or contains invalid keywords
        """
        if not self.is_initialized:
            raise RuntimeError("Search engine has not been initialized")
            
        if not keywords:
            raise ValueError("Keywords list cannot be empty")
            
        # Filter out empty keywords
        valid_keywords = [kw for kw in keywords if kw and kw.strip()]
        if not valid_keywords:
            raise ValueError("No valid keywords provided")
            
        # Use default top_k if not provided
        if top_k is None:
            top_k = self.config.top_k_results
        
        try:
            logger.info(f"在字段 '{field}' 中搜索关键词 {valid_keywords} (匹配所有={match_all})")
            
            # Search for QA data containing the keywords
            matching_qa_data = self.data_loader.search_by_keywords(valid_keywords, field, match_all)
            
            # Convert to search results
            results = []
            for rank, qa_data in enumerate(matching_qa_data[:top_k], 1):
                # Calculate match score based on how many keywords match
                matched_keywords = []
                for keyword in valid_keywords:
                    if qa_data.has_keyword(keyword, field):
                        matched_keywords.append(keyword)
                
                match_ratio = len(matched_keywords) / len(valid_keywords)
                field_boost = 1.0 + (match_ratio * 0.5)  # Boost based on match ratio
                
                similarity_score = SimilarityScore(
                    tfidf_score=field_boost,
                    jaccard_score=match_ratio,
                    bm25_score=field_boost,
                    final_score=field_boost,
                    match_keywords=matched_keywords
                )
                
                result = SearchResult(qa_data, similarity_score, rank)
                results.append(result)
            
            logger.info(f"在字段 '{field}' 中找到 {len(results)} 条匹配关键词 {valid_keywords} 的结果")
            return results
            
        except Exception as e:
            logger.error(f"多关键词搜索过程中出错: {e}")
            raise
    
    def get_qa_by_id(self, qa_id: str) -> Optional[QAData]:
        """
        Get QA data by ID.
        
        Args:
            qa_id: ID of the QA data to retrieve
            
        Returns:
            Optional[QAData]: QA data object if found, None otherwise
            
        Raises:
            RuntimeError: If the search engine has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("Search engine has not been initialized")
            
        return self.data_loader.get_qa_by_id(qa_id)
    
    def get_all_qa_data(self) -> List[QAData]:
        """
        Get all loaded QA data.
        
        Returns:
            List[QAData]: List of all QA data objects
            
        Raises:
            RuntimeError: If the search engine has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("Search engine has not been initialized")
            
        return self.data_loader.get_all_qa_data()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get search engine statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing statistics
            
        Raises:
            RuntimeError: If the search engine has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("Search engine has not been initialized")
            
        try:
            stats = {
                "total_qa_pairs": self.data_loader.get_qa_count(),
                "total_stopwords": self.stopwords_manager.get_stopword_count(),
                "total_custom_stopwords": len(self.stopwords_manager.get_custom_stopwords()),
                "similarity_index_built": self.similarity_calculator.is_index_built(),
                "similarity_threshold": self.config.similarity_threshold,
                "default_top_k": self.config.top_k_results,
                "max_query_length": self.config.max_query_length
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息时出错: {e}")
            raise
    
    def add_custom_stopwords(self, words: List[str]) -> None:
        """
        Add custom stopwords.
        
        Args:
            words: List of custom stopwords to add
            
        Raises:
            RuntimeError: If the search engine has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("Search engine has not been initialized")
            
        try:
            self.stopwords_manager.add_custom_stopwords(words)
            logger.info(f"添加了 {len(words)} 个自定义停用词")
            
        except Exception as e:
            logger.error(f"添加自定义停用词时出错: {e}")
            raise
    
    def remove_custom_stopwords(self, words: List[str]) -> None:
        """
        Remove custom stopwords.
        
        Args:
            words: List of custom stopwords to remove
            
        Raises:
            RuntimeError: If the search engine has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("Search engine has not been initialized")
            
        try:
            self.stopwords_manager.remove_stopwords(words)
            logger.info(f"移除了 {len(words)} 个自定义停用词")
            
        except Exception as e:
            logger.error(f"移除自定义停用词时出错: {e}")
            raise
    
    def reload_data(self) -> None:
        """
        Reload data and rebuild indexes.
        
        Raises:
            RuntimeError: If the search engine has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("Search engine has not been initialized")
            
        try:
            logger.info("重新加载数据并重建索引")
            
            # Reload data
            self.data_loader.reload_data()
            self.qa_data_list = self.data_loader.get_all_qa_data()
            
            # Rebuild similarity index with combined text
            combined_texts = []
            for qa in self.qa_data_list:
                combined_text = f"{qa.question} {qa.answer}"
                combined_texts.append(combined_text)
            
            self.similarity_calculator.rebuild_index(combined_texts)
            
            logger.info("数据重新加载完成，索引重建成功")
            
        except Exception as e:
            logger.error(f"重新加载数据时出错: {e}")
            raise
    
    def is_initialized(self) -> bool:
        """
        Check if the search engine has been initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self.is_initialized