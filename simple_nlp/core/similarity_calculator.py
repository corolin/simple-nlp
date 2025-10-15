"""
Similarity calculation module for the NLP search system.
"""
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from simple_nlp.core.nlp_processor import NLPProcessor
from simple_nlp.utils.text_utils import TextUtils

logger = logging.getLogger(__name__)


@dataclass
class SimilarityScore:
    """
    Data class to store similarity scores.
    
    Attributes:
        tfidf_score: TF-IDF similarity score
        jaccard_score: Jaccard similarity score
        bm25_score: BM25 similarity score
        final_score: Final combined similarity score
        match_keywords: List of matching keywords
    """
    tfidf_score: float = 0.0
    jaccard_score: float = 0.0
    bm25_score: float = 0.0
    final_score: float = 0.0
    match_keywords: List[str] = None
    
    def __post_init__(self):
        """初始化默认值。"""
        if self.match_keywords is None:
            self.match_keywords = []


class SimilarityCalculator:
    """
    Calculator for text similarity using multiple algorithms.
    
    This class is responsible for:
    - Text vectorization representation
    - Similarity algorithm implementation
    - Relevance score calculation
    """
    
    def __init__(self, nlp_processor: NLPProcessor):
        """
        Initialize the SimilarityCalculator with an NLP processor.
        
        Args:
            nlp_processor: NLP processor for text preprocessing
        """
        self.nlp_processor = nlp_processor
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.processed_documents: List[str] = []
        self.tokenized_documents: List[List[str]] = []
        self.is_initialized = False
    
    def build_index(self, documents: List[str]) -> None:
        """
        Build similarity index from a list of documents.
        
        Args:
            documents: List of document texts
            
        Raises:
            RuntimeError: If the NLP processor has not been initialized
        """
        if not self.nlp_processor.is_initialized:
            raise RuntimeError("NLP processor has not been initialized")
            
        if not documents:
            logger.warning("未提供用于构建索引的文档")
            return
        
        try:
            logger.info(f"为 {len(documents)} 个文档构建相似度索引")
            
            # Preprocess documents
            self.processed_documents = []
            self.tokenized_documents = []
            
            for doc in documents:
                # Preprocess the document
                processed_doc = self.nlp_processor.preprocess_text(doc)
                self.processed_documents.append(processed_doc)
                
                # Tokenize the document for BM25
                # First normalize the document text to lowercase to ensure case insensitivity
                normalized_doc = TextUtils.normalize_text(doc)
                tokens = self.nlp_processor.segment(normalized_doc)
                filtered_tokens = self.nlp_processor.remove_stopwords(tokens)
                self.tokenized_documents.append(filtered_tokens)
            
            # Build TF-IDF index
            self._build_tfidf_index()
            
            # Build BM25 index
            self._build_bm25_index()
            
            self.is_initialized = True
            logger.info("相似度索引构建成功")
            
        except Exception as e:
            logger.error(f"构建相似度索引时出错: {e}")
            raise
    
    def _build_tfidf_index(self) -> None:
        """从处理过的文档构建TF-IDF索引。"""
        try:
            # 创建TF-IDF向量化器
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words=None,  # 我们已经移除了停用词
                ngram_range=(1, 2),
                sublinear_tf=True
            )
            
            # 拟合和转换文档
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_documents)
            
            logger.info(f"TF-IDF索引构建完成，包含 {self.tfidf_matrix.shape[1]} 个特征")
            
        except Exception as e:
            logger.error(f"构建TF-IDF索引时出错: {e}")
            raise
    
    def _build_bm25_index(self) -> None:
        """从标记化文档构建BM25索引。"""
        try:
            # 创建BM25索引
            self.bm25_index = BM25Okapi(self.tokenized_documents)
            
            logger.info("BM25索引构建成功")
            
        except Exception as e:
            logger.error(f"构建BM25索引时出错: {e}")
            raise
    
    def calculate_similarity(self, query: str, doc_index: int) -> SimilarityScore:
        """
        Calculate similarity between a query and a document.
        
        Args:
            query: Query text
            doc_index: Index of the document in the index
            
        Returns:
            SimilarityScore: Similarity scores
            
        Raises:
            RuntimeError: If the index has not been built
        """
        if not self.is_initialized:
            raise RuntimeError("Similarity index has not been built")
            
        if doc_index < 0 or doc_index >= len(self.processed_documents):
            raise ValueError(f"Invalid document index: {doc_index}")
        
        try:
            # Preprocess query
            processed_query = self.nlp_processor.preprocess_text(query)
            
            # Calculate TF-IDF similarity
            tfidf_score = self._calculate_tfidf_similarity(processed_query, doc_index)
            
            # Calculate Jaccard similarity
            jaccard_score = self._calculate_jaccard_similarity(processed_query, doc_index)
            
            # Calculate BM25 similarity
            bm25_score = self._calculate_bm25_similarity(query, doc_index)
            
            # Calculate final score (weighted average)
            final_score = self._calculate_final_score(tfidf_score, jaccard_score, bm25_score)
            
            # Find matching keywords
            match_keywords = self._find_matching_keywords(processed_query, doc_index)
            
            return SimilarityScore(
                tfidf_score=tfidf_score,
                jaccard_score=jaccard_score,
                bm25_score=bm25_score,
                final_score=final_score,
                match_keywords=match_keywords
            )
            
        except Exception as e:
            logger.error(f"计算相似度时出错: {e}")
            raise
    
    def _calculate_tfidf_similarity(self, processed_query: str, doc_index: int) -> float:
        """计算查询和文档之间的TF-IDF相似度。"""
        try:
            # 将查询转换为TF-IDF向量
            query_vector = self.tfidf_vectorizer.transform([processed_query])
            
            # 获取文档向量
            doc_vector = self.tfidf_matrix[doc_index]
            
            # 计算余弦相似度
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算TF-IDF相似度时出错: {e}")
            return 0.0
    
    def _calculate_jaccard_similarity(self, processed_query: str, doc_index: int) -> float:
        """计算查询和文档之间的Jaccard相似度。"""
        try:
            # DEBUG: 添加日志记录
            logger.debug(f"Jaccard - 处理后的查询: '{processed_query}'")
            logger.debug(f"Jaccard - 文档[{doc_index}]: '{self.processed_documents[doc_index]}'")
            
            # 获取查询和文档标记
            query_tokens = set(processed_query.split())
            doc_tokens = set(self.processed_documents[doc_index].split())
            
            logger.debug(f"Jaccard - 查询标记: {query_tokens}")
            logger.debug(f"Jaccard - 文档标记: {doc_tokens}")
            
            if not query_tokens or not doc_tokens:
                return 0.0
            
            # 计算Jaccard相似度
            intersection = query_tokens.intersection(doc_tokens)
            union = query_tokens.union(doc_tokens)
            
            logger.debug(f"Jaccard - 交集: {intersection}")
            logger.debug(f"Jaccard - 并集: {union}")
            
            similarity = len(intersection) / len(union)
            logger.debug(f"Jaccard - 相似度: {similarity}")
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算Jaccard相似度时出错: {e}")
            return 0.0
    
    def _calculate_bm25_similarity(self, query: str, doc_index: int) -> float:
        """计算查询和文档之间的BM25相似度。"""
        try:
            # 标记化查询
            # 首先将查询文本标准化为小写以确保大小写不敏感
            normalized_query = TextUtils.normalize_text(query)
            query_tokens = self.nlp_processor.segment(normalized_query)
            filtered_query_tokens = self.nlp_processor.remove_stopwords(query_tokens)
            
            # 计算BM25分数
            bm25_scores = self.bm25_index.get_scores(filtered_query_tokens)
            
            return float(bm25_scores[doc_index])
            
        except Exception as e:
            logger.error(f"计算BM25相似度时出错: {e}")
            return 0.0
    
    def _calculate_final_score(self, tfidf_score: float, jaccard_score: float, bm25_score: float) -> float:
        """计算最终相似度分数作为加权平均值。"""
        # 不同相似度度量的权重
        tfidf_weight = 0.5
        jaccard_weight = 0.2
        bm25_weight = 0.3
        
        # 将分数标准化到[0, 1]范围
        normalized_tfidf = min(max(tfidf_score, 0.0), 1.0)
        normalized_jaccard = min(max(jaccard_score, 0.0), 1.0)
        
        # 标准化BM25分数（BM25可以大于1的值）
        # 我们将使用类似sigmoid的函数来标准化
        normalized_bm25 = 1.0 - 1.0 / (1.0 + bm25_score)
        
        # 计算加权平均值
        final_score = (
            tfidf_weight * normalized_tfidf +
            jaccard_weight * normalized_jaccard +
            bm25_weight * normalized_bm25
        )
        
        return float(final_score)
    
    def _find_matching_keywords(self, processed_query: str, doc_index: int) -> List[str]:
        """查找查询和文档之间匹配的关键词。"""
        try:
            # 获取查询和文档标记
            query_tokens = set(processed_query.split())
            doc_tokens = set(self.processed_documents[doc_index].split())
            
            # 找到交集
            matching_tokens = query_tokens.intersection(doc_tokens)
            
            # 按长度过滤并排序
            matching_keywords = [
                token for token in matching_tokens 
                if len(token) >= 2  # 只考虑至少2个字符的标记
            ]
            
            return sorted(matching_keywords)
            
        except Exception as e:
            logger.error(f"查找匹配关键词时出错: {e}")
            return []
    
    def search_similar_documents(self, query: str, top_k: int = 5) -> List[Tuple[int, SimilarityScore]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List[Tuple[int, SimilarityScore]]: List of (document_index, similarity_score) tuples
            
        Raises:
            RuntimeError: If the index has not been built
        """
        if not self.is_initialized:
            raise RuntimeError("Similarity index has not been built")
            
        if not query:
            return []
        
        try:
            logger.info(f"搜索与以下内容相似的文档: {query}")
            
            # Calculate similarity for all documents
            similarities = []
            for doc_index in range(len(self.processed_documents)):
                similarity_score = self.calculate_similarity(query, doc_index)
                similarities.append((doc_index, similarity_score))
            
            # Sort by final score in descending order
            similarities.sort(key=lambda x: x[1].final_score, reverse=True)
            
            # Return top_k results
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"搜索相似文档时出错: {e}")
            raise
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the index.
        
        Returns:
            int: Number of documents
        """
        return len(self.processed_documents)
    
    def is_index_built(self) -> bool:
        """
        Check if the similarity index has been built.
        
        Returns:
            bool: True if the index is built, False otherwise
        """
        return self.is_initialized
    
    def rebuild_index(self, documents: List[str]) -> None:
        """
        Rebuild the similarity index with new documents.
        
        Args:
            documents: List of document texts
        """
        logger.info("重新构建相似度索引")
        self.build_index(documents)