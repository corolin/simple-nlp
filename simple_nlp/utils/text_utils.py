"""
Text utility functions for the NLP search system.
"""
import re
import logging
from typing import List, Set, Optional

logger = logging.getLogger(__name__)


class TextUtils:
    """文本处理操作的工具类。"""
    
    # Common punctuation and special characters to remove
    PUNCTUATION_PATTERN = re.compile(r'[^\w\s\u4e00-\u9fff]')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing special characters and normalizing whitespace.
        
        Args:
            text: Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Remove special characters but keep Chinese characters, letters, numbers, and whitespace
        cleaned = TextUtils.PUNCTUATION_PATTERN.sub(' ', text)
        
        # Normalize whitespace
        cleaned = TextUtils.WHITESPACE_PATTERN.sub(' ', cleaned).strip()
        
        return cleaned
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text by converting to lowercase and removing extra whitespace.
        
        Args:
            text: Input text to normalize
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = TextUtils.WHITESPACE_PATTERN.sub(' ', normalized).strip()
        
        return normalized
    
    @staticmethod
    def extract_chinese_words(text: str) -> List[str]:
        """
        Extract Chinese words from text.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: List of Chinese words
        """
        if not text:
            return []
            
        # Match Chinese characters (Unicode range for common Chinese characters)
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        chinese_words = chinese_pattern.findall(text)
        
        return chinese_words
    
    @staticmethod
    def filter_by_length(tokens: List[str], min_length: int = 1, max_length: Optional[int] = None) -> List[str]:
        """
        Filter tokens by length.
        
        Args:
            tokens: List of tokens to filter
            min_length: Minimum token length
            max_length: Maximum token length (None for no limit)
            
        Returns:
            List[str]: Filtered tokens
        """
        if not tokens:
            return []
            
        filtered = []
        for token in tokens:
            token_len = len(token)
            if token_len >= min_length and (max_length is None or token_len <= max_length):
                filtered.append(token)
                
        return filtered
    
    @staticmethod
    def remove_duplicates(tokens: List[str]) -> List[str]:
        """
        Remove duplicate tokens while preserving order.
        
        Args:
            tokens: List of tokens that may contain duplicates
            
        Returns:
            List[str]: List of tokens with duplicates removed
        """
        if not tokens:
            return []
            
        seen = set()
        unique_tokens = []
        
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
                
        return unique_tokens
    
    @staticmethod
    def count_words(text: str) -> int:
        """
        Count the number of words in text.
        
        Args:
            text: Input text
            
        Returns:
            int: Number of words
        """
        if not text:
            return 0
            
        # Split by whitespace and filter out empty strings
        words = text.split()
        return len(words)
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to a maximum length.
        
        Args:
            text: Input text
            max_length: Maximum length of the truncated text
            suffix: Suffix to add if text is truncated
            
        Returns:
            str: Truncated text
        """
        if not text:
            return ""
            
        if len(text) <= max_length:
            return text
            
        # Reserve space for suffix
        if len(suffix) >= max_length:
            return suffix[:max_length]
            
        truncated = text[:max_length - len(suffix)] + suffix
        return truncated
    
    @staticmethod
    def is_chinese_char(char: str) -> bool:
        """
        Check if a character is a Chinese character.
        
        Args:
            char: Character to check
            
        Returns:
            bool: True if the character is Chinese, False otherwise
        """
        if not char or len(char) != 1:
            return False
            
        # Check if character is in the Chinese Unicode range
        return '\u4e00' <= char <= '\u9fff'
    
    @staticmethod
    def is_punctuation(char: str) -> bool:
        """
        Check if a character is punctuation.
        
        Args:
            char: Character to check
            
        Returns:
            bool: True if the character is punctuation, False otherwise
        """
        if not char or len(char) != 1:
            return False
            
        # Check if character is a punctuation mark
        return char in '，。！？；：""''（）【】《》、'
    
    @staticmethod
    def contains_chinese(text: str) -> bool:
        """
        Check if text contains Chinese characters.
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if text contains Chinese characters, False otherwise
        """
        if not text:
            return False
            
        return any(TextUtils.is_chinese_char(char) for char in text)
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split text into sentences based on Chinese and English punctuation.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: List of sentences
        """
        if not text:
            return []
            
        # Split by sentence-ending punctuation
        sentences = re.split(r'[。！？.!?]+', text)
        
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    @staticmethod
    def merge_tokens(tokens: List[str], separator: str = " ") -> str:
        """
        Merge tokens into a single string.
        
        Args:
            tokens: List of tokens to merge
            separator: Separator to use between tokens
            
        Returns:
            str: Merged string
        """
        if not tokens:
            return ""
            
        return separator.join(tokens)
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two texts based on word overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
            
        # Normalize and split texts into words
        words1 = set(TextUtils.normalize_text(text1).split())
        words2 = set(TextUtils.normalize_text(text2).split())
        
        if not words1 or not words2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)