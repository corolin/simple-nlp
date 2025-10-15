"""
NLP processing module for the NLP search system.
"""
import logging
from typing import List, Optional, Set

import jieba
from jieba import analyse

from simple_nlp.core.stopwords_manager import StopWordsManager
from simple_nlp.utils.text_utils import TextUtils

logger = logging.getLogger(__name__)


class NLPProcessor:
    """
    NLP processor for text processing operations.
    
    This class is responsible for:
    - Chinese text segmentation
    - Stopword filtering
    - Text standardization
    - Keyword extraction
    """
    
    def __init__(self, stopwords_manager: StopWordsManager):
        """
        Initialize the NLPProcessor with a stopwords manager.
        
        Args:
            stopwords_manager: Manager for handling stopwords
        """
        self.stopwords_manager = stopwords_manager
        self.is_initialized = False
        
        # Initialize jieba
        self._initialize_jieba()
    
    def _initialize_jieba(self) -> None:
        """使用自定义设置初始化jieba。"""
        try:
            # 设置jieba使用最精确的分词模式
            jieba.initialize()
            
            # 配置jieba分析器用于关键词提取
            # 我们将自己处理停用词，所以不设置stop_words文件
            
            self.is_initialized = True
            logger.info("NLP处理器初始化成功")
            
        except Exception as e:
            logger.error(f"NLP处理器初始化失败: {e}")
            raise
    
    def segment(self, text: str, cut_all: bool = False, HMM: bool = True) -> List[str]:
        """
        Segment Chinese text into words.
        
        Args:
            text: Input text to segment
            cut_all: Whether to use full segmentation mode
            HMM: Whether to use HMM for unknown words
            
        Returns:
            List[str]: List of segmented words
            
        Raises:
            RuntimeError: If the processor has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("NLP processor has not been initialized")
            
        if not text:
            return []
        
        try:
            # 使用jieba进行分词
            words = jieba.cut(text, cut_all=cut_all, HMM=HMM)
            
            # 转换为列表并过滤空字符串
            words = [word.strip() for word in words if word.strip()]
            
            return words
            
        except Exception as e:
            logger.error(f"文本分词过程中出错: {e}")
            raise
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from a list of tokens.
        
        Args:
            tokens: List of tokens to filter
            
        Returns:
            List[str]: List of tokens with stopwords removed
            
        Raises:
            RuntimeError: If the processor has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("NLP processor has not been initialized")
            
        if not tokens:
            return []
        
        try:
            # 使用停用词管理器过滤标记
            filtered_tokens = self.stopwords_manager.filter_stopwords(tokens)
            
            return filtered_tokens
            
        except Exception as e:
            logger.error(f"移除停用词过程中出错: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning, segmenting, and removing stopwords.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            str: Preprocessed text
            
        Raises:
            RuntimeError: If the processor has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("NLP processor has not been initialized")
            
        if not text:
            return ""
        
        try:
            # DEBUG: 记录原始文本的日志
            logger.debug(f"原始文本: '{text}'")
            
            # 清理文本
            cleaned_text = TextUtils.clean_text(text)
            logger.debug(f"清理后的文本: '{cleaned_text}'")
            
            # 将文本标准化为小写（修复大小写敏感问题）
            normalized_text = TextUtils.normalize_text(cleaned_text)
            logger.debug(f"标准化后的文本: '{normalized_text}'")
            
            # 分词文本
            tokens = self.segment(normalized_text)
            logger.debug(f"分词结果: {tokens}")
            
            # 移除停用词
            filtered_tokens = self.remove_stopwords(tokens)
            logger.debug(f"过滤后的标记: {filtered_tokens}")
            
            # 将标记重新连接成字符串
            processed_text = " ".join(filtered_tokens)
            logger.debug(f"最终处理后的文本: '{processed_text}'")
            
            return processed_text
            
        except Exception as e:
            logger.error(f"文本预处理过程中出错: {e}")
            raise
    
    def extract_keywords(self, text: str, top_k: int = 10, with_weight: bool = False) -> List[str]:
        """
        Extract keywords from text using TF-IDF algorithm.
        
        Args:
            text: Input text to extract keywords from
            top_k: Number of top keywords to return
            with_weight: Whether to return keywords with weights
            
        Returns:
            List[str]: List of keywords
            
        Raises:
            RuntimeError: If the processor has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("NLP processor has not been initialized")
            
        if not text:
            return []
        
        try:
            # 使用TF-IDF提取关键词
            if with_weight:
                keywords_with_weights = analyse.extract_tags(text, topK=top_k, withWeight=True)
                keywords = [keyword for keyword, weight in keywords_with_weights]
            else:
                keywords = analyse.extract_tags(text, topK=top_k)
            
            return keywords
            
        except Exception as e:
            logger.error(f"关键词提取过程中出错: {e}")
            raise
    
    def extract_keywords_textrank(self, text: str, top_k: int = 10, with_weight: bool = False) -> List[str]:
        """
        Extract keywords from text using TextRank algorithm.
        
        Args:
            text: Input text to extract keywords from
            top_k: Number of top keywords to return
            with_weight: Whether to return keywords with weights
            
        Returns:
            List[str]: List of keywords
            
        Raises:
            RuntimeError: If the processor has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("NLP processor has not been initialized")
            
        if not text:
            return []
        
        try:
            # 使用TextRank提取关键词
            if with_weight:
                keywords_with_weights = analyse.textrank(text, topK=top_k, withWeight=True)
                keywords = [keyword for keyword, weight in keywords_with_weights]
            else:
                keywords = analyse.textrank(text, topK=top_k)
            
            return keywords
            
        except Exception as e:
            logger.error(f"TextRank关键词提取过程中出错: {e}")
            raise
    
    def get_word_frequency(self, text: str) -> dict:
        """
        Get word frequency statistics from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict: Dictionary with words as keys and frequencies as values
            
        Raises:
            RuntimeError: If the processor has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("NLP processor has not been initialized")
            
        if not text:
            return {}
        
        try:
            # 分词文本
            tokens = self.segment(text)
            
            # 移除停用词
            filtered_tokens = self.remove_stopwords(tokens)
            
            # 统计词频
            word_freq = {}
            for word in filtered_tokens:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            return word_freq
            
        except Exception as e:
            logger.error(f"词频分析过程中出错: {e}")
            raise
    
    def get_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts based on word overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score between 0 and 1
            
        Raises:
            RuntimeError: If the processor has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("NLP processor has not been initialized")
            
        if not text1 or not text2:
            return 0.0
        
        try:
            # 预处理两个文本
            processed_text1 = self.preprocess_text(text1)
            processed_text2 = self.preprocess_text(text2)
            
            # 使用TextUtils计算相似度
            similarity = TextUtils.calculate_text_similarity(processed_text1, processed_text2)
            
            return similarity
            
        except Exception as e:
            logger.error(f"文本相似度计算过程中出错: {e}")
            raise
    
    def add_custom_word(self, word: str, freq: Optional[int] = None, tag: Optional[str] = None) -> None:
        """
        Add a custom word to jieba's dictionary.
        
        Args:
            word: Word to add
            freq: Word frequency (None for auto)
            tag: Word tag (None for auto)
            
        Raises:
            RuntimeError: If the processor has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("NLP processor has not been initialized")
            
        try:
            jieba.add_word(word, freq=freq, tag=tag)
            logger.info(f"添加了自定义词语: {word}")
            
        except Exception as e:
            logger.error(f"添加自定义词语时出错: {e}")
            raise
    
    def load_user_dict(self, dict_path: str) -> None:
        """
        Load a user dictionary for jieba.
        
        Args:
            dict_path: Path to the user dictionary file
            
        Raises:
            RuntimeError: If the processor has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("NLP processor has not been initialized")
            
        try:
            jieba.load_userdict(dict_path)
            logger.info(f"从 {dict_path} 加载了用户词典")
            
        except Exception as e:
            logger.error(f"加载用户词典时出错: {e}")
            raise
    
    def delete_word(self, word: str) -> None:
        """
        Delete a word from jieba's dictionary.
        
        Args:
            word: Word to delete
            
        Raises:
            RuntimeError: If the processor has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("NLP processor has not been initialized")
            
        try:
            jieba.del_word(word)
            logger.info(f"删除了词语: {word}")
            
        except Exception as e:
            logger.error(f"删除词语时出错: {e}")
            raise
    
    def suggest_words(self, sentence: str, top_k: int = 5) -> List[tuple]:
        """
        Get word suggestions for a sentence.
        
        Args:
            sentence: Input sentence
            top_k: Number of suggestions to return
            
        Returns:
            List[tuple]: List of (word, frequency) tuples
            
        Raises:
            RuntimeError: If the processor has not been initialized
        """
        if not self.is_initialized:
            raise RuntimeError("NLP processor has not been initialized")
            
        if not sentence:
            return []
        
        try:
            # Get word suggestions
            suggestions = jieba.suggest_freq((sentence,), tune=True)
            
            # Convert to list of tuples and return top_k
            word_freq_pairs = [(word, freq) for word, freq in suggestions.items()]
            word_freq_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return word_freq_pairs[:top_k]
            
        except Exception as e:
            logger.error(f"获取词语建议时出错: {e}")
            raise