"""
Stopwords management module for the NLP search system.
"""
import glob
import logging
import os
from typing import Set, List, Optional

from simple_nlp.utils.file_utils import FileUtils

logger = logging.getLogger(__name__)


class StopWordsManager:
    """
    Manager for handling stopwords in the NLP search system.
    
    This class is responsible for:
    - Loading stopwords from file
    - Managing stopwords collection
    - Supporting custom stopwords extension
    """
    
    def __init__(self, stopwords_file: str):
        """
        Initialize the StopWordsManager with the path to the stopwords file.
        
        Args:
            stopwords_file: Path to the stopwords file
        """
        self.stopwords_file = stopwords_file
        self.stop_words: Set[str] = set()
        self.custom_stop_words: Set[str] = set()
        self.is_loaded = False
        
    def load_stopwords(self) -> None:
        """
        从文件或目录加载停用词。
        
        支持单个文件或包含多个txt文件的目录。
        
        异常:
            FileNotFoundError: 如果停用词文件或目录不存在
            UnicodeDecodeError: 如果文件无法解码
        """
        logger.info(f"从 {self.stopwords_file} 加载停用词")
        
        try:
            self.stop_words = set()
            
            if os.path.isdir(self.stopwords_file):
                # 如果是目录，读取所有txt文件
                self._load_stopwords_from_directory()
            else:
                # 如果是文件，读取单个文件
                self._load_stopwords_from_file(self.stopwords_file)
            
            self.is_loaded = True
            logger.info(f"成功加载 {len(self.stop_words)} 个停用词")
            
        except Exception as e:
            logger.error(f"加载停用词失败: {e}")
            raise
    
    def _load_stopwords_from_directory(self) -> None:
        """
        从目录中加载所有txt文件的停用词。
        """
        logger.info(f"从目录加载停用词: {self.stopwords_file}")
        
        # 获取目录下所有txt文件
        txt_files = glob.glob(os.path.join(self.stopwords_file, "*.txt"))
        
        if not txt_files:
            logger.warning(f"在目录中未找到txt文件: {self.stopwords_file}")
            return
        
        for txt_file in sorted(txt_files):
            logger.info(f"从文件加载停用词: {txt_file}")
            self._load_stopwords_from_file(txt_file)
    
    def _load_stopwords_from_file(self, file_path: str) -> None:
        """
        从单个文件加载停用词。
        
        参数:
            file_path: 停用词文件路径
        """
        try:
            # 读取停用词文件
            lines = FileUtils.read_text_file_as_lines(file_path)
            
            # 处理每一行
            file_stopwords = set()
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):  # 跳过空行和注释
                    file_stopwords.add(line)
            
            self.stop_words.update(file_stopwords)
            logger.info(f"从 {os.path.basename(file_path)} 加载了 {len(file_stopwords)} 个停用词")
            
        except Exception as e:
            logger.error(f"从 {file_path} 加载停用词失败: {e}")
            raise
    
    def add_custom_stopwords(self, words: List[str]) -> None:
        """
        Add custom stopwords to the existing collection.
        
        Args:
            words: List of custom stopwords to add
            
        Raises:
            RuntimeError: If stopwords have not been loaded yet
        """
        if not self.is_loaded:
            raise RuntimeError("Stopwords have not been loaded yet. Call load_stopwords() first.")
            
        for word in words:
            word = word.strip()
            if word:
                self.custom_stop_words.add(word)
        
        logger.info(f"添加了 {len(words)} 个自定义停用词")
    
    def remove_stopwords(self, words: List[str]) -> None:
        """
        Remove stopwords from the existing collection.
        
        Args:
            words: List of stopwords to remove
            
        Raises:
            RuntimeError: If stopwords have not been loaded yet
        """
        if not self.is_loaded:
            raise RuntimeError("Stopwords have not been loaded yet. Call load_stopwords() first.")
            
        removed_count = 0
        for word in words:
            word = word.strip()
            if word in self.stop_words:
                self.stop_words.remove(word)
                removed_count += 1
            if word in self.custom_stop_words:
                self.custom_stop_words.remove(word)
                removed_count += 1
        
        logger.info(f"移除了 {removed_count} 个停用词")
    
    def is_stopword(self, word: str) -> bool:
        """
        Check if a word is a stopword.
        
        Args:
            word: Word to check
            
        Returns:
            bool: True if the word is a stopword, False otherwise
            
        Raises:
            RuntimeError: If stopwords have not been loaded yet
        """
        if not self.is_loaded:
            raise RuntimeError("Stopwords have not been loaded yet. Call load_stopwords() first.")
            
        # DEBUG: 添加日志记录
        logger.debug(f"检查 '{word}' 是否为停用词")
        
        # 修复大小写敏感问题：统一转换为小写进行比较
        word_lower = word.lower()
        is_stop = word_lower in self.stop_words or word_lower in self.custom_stop_words
        
        logger.debug(f"词语 '{word}' (小写: '{word_lower}') 是停用词: {is_stop}")
        
        return is_stop
    
    def filter_stopwords(self, words: List[str]) -> List[str]:
        """
        Filter stopwords from a list of words.
        
        Args:
            words: List of words to filter
            
        Returns:
            List[str]: List of words with stopwords removed
            
        Raises:
            RuntimeError: If stopwords have not been loaded yet
        """
        if not self.is_loaded:
            raise RuntimeError("Stopwords have not been loaded yet. Call load_stopwords() first.")
            
        return [word for word in words if not self.is_stopword(word)]
    
    def get_all_stopwords(self) -> Set[str]:
        """
        Get all stopwords including custom ones.
        
        Returns:
            Set[str]: Set of all stopwords
            
        Raises:
            RuntimeError: If stopwords have not been loaded yet
        """
        if not self.is_loaded:
            raise RuntimeError("Stopwords have not been loaded yet. Call load_stopwords() first.")
            
        return self.stop_words.union(self.custom_stop_words)
    
    def get_default_stopwords(self) -> Set[str]:
        """
        Get the default stopwords (excluding custom ones).
        
        Returns:
            Set[str]: Set of default stopwords
            
        Raises:
            RuntimeError: If stopwords have not been loaded yet
        """
        if not self.is_loaded:
            raise RuntimeError("Stopwords have not been loaded yet. Call load_stopwords() first.")
            
        return self.stop_words.copy()
    
    def get_custom_stopwords(self) -> Set[str]:
        """
        Get the custom stopwords.
        
        Returns:
            Set[str]: Set of custom stopwords
            
        Raises:
            RuntimeError: If stopwords have not been loaded yet
        """
        if not self.is_loaded:
            raise RuntimeError("Stopwords have not been loaded yet. Call load_stopwords() first.")
            
        return self.custom_stop_words.copy()
    
    def get_stopword_count(self) -> int:
        """
        Get the total number of stopwords.
        
        Returns:
            int: Number of stopwords
            
        Raises:
            RuntimeError: If stopwords have not been loaded yet
        """
        if not self.is_loaded:
            raise RuntimeError("Stopwords have not been loaded yet. Call load_stopwords() first.")
            
        return len(self.stop_words) + len(self.custom_stop_words)
    
    def clear_custom_stopwords(self) -> None:
        """
        Clear all custom stopwords.
        
        Raises:
            RuntimeError: If stopwords have not been loaded yet
        """
        if not self.is_loaded:
            raise RuntimeError("Stopwords have not been loaded yet. Call load_stopwords() first.")
            
        self.custom_stop_words.clear()
        logger.info("清除了所有自定义停用词")
    
    def save_custom_stopwords(self, file_path: str) -> None:
        """
        Save custom stopwords to a file.
        
        Args:
            file_path: Path to save the custom stopwords
            
        Raises:
            RuntimeError: If stopwords have not been loaded yet
            IOError: If the file cannot be written
        """
        if not self.is_loaded:
            raise RuntimeError("Stopwords have not been loaded yet. Call load_stopwords() first.")
            
        try:
            # Sort stopwords for consistent output
            sorted_stopwords = sorted(self.custom_stop_words)
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                for word in sorted_stopwords:
                    f.write(word + '\n')
            
            logger.info(f"保存了 {len(self.custom_stop_words)} 个自定义停用词到 {file_path}")
            
        except Exception as e:
            logger.error(f"保存自定义停用词失败: {e}")
            raise
    
    def load_custom_stopwords(self, file_path: str) -> None:
        """
        Load custom stopwords from a file.
        
        Args:
            file_path: Path to the custom stopwords file
            
        Raises:
            RuntimeError: If stopwords have not been loaded yet
            FileNotFoundError: If the file does not exist
            UnicodeDecodeError: If the file cannot be decoded
        """
        if not self.is_loaded:
            raise RuntimeError("Stopwords have not been loaded yet. Call load_stopwords() first.")
            
        try:
            # Read the custom stopwords file
            lines = FileUtils.read_text_file_as_lines(file_path)
            
            # Process each line
            custom_words = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    custom_words.append(line)
            
            # Add to custom stopwords
            self.add_custom_stopwords(custom_words)
            
            logger.info(f"从 {file_path} 加载了 {len(custom_words)} 个自定义停用词")
            
        except Exception as e:
            logger.error(f"加载自定义停用词失败: {e}")
            raise
    
    def reload_stopwords(self) -> None:
        """
        Reload the stopwords from the file.
        
        This is useful when the stopwords file has been updated.
        """
        logger.info("重新加载停用词")
        self.stop_words.clear()
        self.custom_stop_words.clear()
        self.load_stopwords()
    
    def is_loaded(self) -> bool:
        """
        Check if stopwords have been loaded.
        
        Returns:
            bool: True if stopwords are loaded, False otherwise
        """
        return self.is_loaded