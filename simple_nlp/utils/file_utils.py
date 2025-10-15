"""
File utility functions for the NLP search system.
"""
import json
import os
import logging
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


class FileUtils:
    """文件操作的工具类。"""
    
    @staticmethod
    def read_jsonl(file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Read a JSONL file and yield each line as a dictionary.
        
        Args:
            file_path: Path to the JSONL file
            
        Yields:
            Dict[str, Any]: Each line parsed as a dictionary
            
        Raises:
            FileNotFoundError: If the file does not exist
            json.JSONDecodeError: If a line cannot be parsed as JSON
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            logger.error(f"解析文件 {file_path} 第 {line_num} 行时出错: {e}")
                            raise
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时出错: {e}")
            raise
    
    @staticmethod
    def read_jsonl_as_list(file_path: str) -> List[Dict[str, Any]]:
        """
        Read a JSONL file and return all lines as a list of dictionaries.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List[Dict[str, Any]]: List of parsed dictionaries
            
        Raises:
            FileNotFoundError: If the file does not exist
            json.JSONDecodeError: If a line cannot be parsed as JSON
        """
        return list(FileUtils.read_jsonl(file_path))
    
    @staticmethod
    def write_jsonl(file_path: str, data: List[Dict[str, Any]], mode: str = 'w') -> None:
        """
        Write a list of dictionaries to a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            data: List of dictionaries to write
            mode: File write mode ('w' for write, 'a' for append)
            
        Raises:
            IOError: If the file cannot be written
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, mode, encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"写入文件 {file_path} 时出错: {e}")
            raise
    
    @staticmethod
    def read_text_file(file_path: str, encoding: str = 'utf-8') -> str:
        """
        Read a text file and return its content as a string.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding
            
        Returns:
            str: Content of the file
            
        Raises:
            FileNotFoundError: If the file does not exist
            UnicodeDecodeError: If the file cannot be decoded with the specified encoding
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError as e:
            logger.error(f"使用编码 {encoding} 解码文件 {file_path} 时出错: {e}")
            raise
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时出错: {e}")
            raise
    
    @staticmethod
    def read_text_file_as_lines(file_path: str, encoding: str = 'utf-8') -> List[str]:
        """
        Read a text file and return its content as a list of lines.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding
            
        Returns:
            List[str]: Lines of the file
            
        Raises:
            FileNotFoundError: If the file does not exist
            UnicodeDecodeError: If the file cannot be decoded with the specified encoding
        """
        content = FileUtils.read_text_file(file_path, encoding)
        return content.splitlines()
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> None:
        """
        Ensure that a directory exists, creating it if necessary.
        
        Args:
            directory_path: Path to the directory
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
        except Exception as e:
            logger.error(f"创建目录 {directory_path} 时出错: {e}")
            raise
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        Get the size of a file in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            int: Size of the file in bytes
            
        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        return os.path.getsize(file_path)
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict[str, Any]: Dictionary containing file information
            
        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        path = Path(file_path)
        stat = path.stat()
        
        return {
            'name': path.name,
            'size': stat.st_size,
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime,
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'extension': path.suffix,
            'absolute_path': str(path.absolute())
        }
    
    @staticmethod
    def is_file_readable(file_path: str) -> bool:
        """
        Check if a file is readable.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if the file is readable, False otherwise
        """
        return os.path.exists(file_path) and os.access(file_path, os.R_OK)
    
    @staticmethod
    def is_file_writable(file_path: str) -> bool:
        """
        Check if a file is writable.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if the file is writable, False otherwise
        """
        if os.path.exists(file_path):
            return os.access(file_path, os.W_OK)
        else:
            # Check if the directory is writable
            directory = os.path.dirname(file_path)
            return os.path.exists(directory) and os.access(directory, os.W_OK)