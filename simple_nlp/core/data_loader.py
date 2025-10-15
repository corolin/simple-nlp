"""
NLP搜索系统的数据加载模块。
"""
import json
import logging
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime

from simple_nlp.models.qa_data import QAData
from simple_nlp.utils.file_utils import FileUtils
from simple_nlp.utils.text_utils import TextUtils

logger = logging.getLogger(__name__)


class DataLoader:
    """
    用于读取和解析JSONL格式QA数据的数据加载器。
    
    此类负责：
    - 读取和解析JSONL格式的QA数据文件
    - 数据验证和清理
    - 构建内存索引结构
    """
    
    def __init__(self, data_file: str):
        """
        使用数据文件路径初始化DataLoader。
        
        参数:
            data_file: JSONL数据文件的路径
        """
        self.data_file = data_file
        self.qa_data_list: List[QAData] = []
        self.id_to_qa: Dict[str, QAData] = {}
        self.is_loaded = False
        
    def load_data(self) -> None:
        """
        加载和解析JSONL数据文件。
        
        异常:
            FileNotFoundError: 如果数据文件不存在
            json.JSONDecodeError: 如果数据文件包含无效的JSON
            ValueError: 如果数据格式无效
        """
        logger.info(f"从 {self.data_file} 加载数据")
        
        try:
            # 读取JSONL文件
            raw_data = FileUtils.read_jsonl_as_list(self.data_file)
            
            # 解析和验证每条记录
            self.qa_data_list = []
            self.id_to_qa = {}
            
            for i, record in enumerate(raw_data):
                try:
                    qa_data = self._parse_qa_record(record)
                    self.qa_data_list.append(qa_data)
                    self.id_to_qa[qa_data.id] = qa_data
                except Exception as e:
                    logger.warning(f"解析第 {i+1} 条记录时出错: {e}")
                    continue
            
            self.is_loaded = True
            logger.info(f"成功加载 {len(self.qa_data_list)} 条QA记录")
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def _parse_qa_record(self, record: Dict[str, Any]) -> QAData:
        """
        从字典中解析单条QA记录。
        
        参数:
            record: 包含QA数据的字典
            
        返回:
            QAData: 解析后的QA数据对象
            
        异常:
            ValueError: 如果记录格式无效
        """
        # 验证必需字段
        if 'id' not in record:
            raise ValueError("Missing required field 'id'")
            
        if 'text' not in record and 'question' not in record:
            raise ValueError("Missing required field 'text' or 'question'")
            
        if 'answer' not in record:
            raise ValueError("Missing required field 'answer'")
        
        # 提取字段，带有回退值
        qa_id = str(record['id'])
        question = record.get('text', record.get('question', ''))
        answer = record.get('answer', '')
        
        # 清理和验证数据
        if not question.strip():
            raise ValueError(f"Question cannot be empty for record {qa_id}")
            
        if not answer.strip():
            raise ValueError(f"Answer cannot be empty for record {qa_id}")
        
        # 从问题和答案中提取关键词
        question_keywords = self._extract_keywords(question)
        answer_keywords = self._extract_keywords(answer)
        
        # 如果可用，解析创建时间
        create_time = None
        if 'create_time' in record:
            try:
                if isinstance(record['create_time'], str):
                    create_time = datetime.fromisoformat(record['create_time'])
                elif isinstance(record['create_time'], (int, float)):
                    create_time = datetime.fromtimestamp(record['create_time'])
            except Exception as e:
                logger.warning(f"解析记录 {qa_id} 的创建时间时出错: {e}")
        
        # 创建QAData对象
        qa_data = QAData(
            id=qa_id,
            question=question.strip(),
            answer=answer.strip(),
            keywords=question_keywords,
            answer_keywords=answer_keywords,
            create_time=create_time or datetime.now()
        )
        
        # 更新组合关键词
        qa_data.update_all_keywords()
        
        return qa_data
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        从文本中提取关键词。
        
        参数:
            text: 输入文本
            
        返回:
            List[str]: 关键词列表
        """
        if not text:
            return []
        
        # DEBUG: 添加日志记录
        logger.debug(f"从文本中提取关键词: '{text}'")
        
        # 清理和标准化文本（修复大小写敏感问题）
        cleaned_text = TextUtils.clean_text(text)
        logger.debug(f"清理后的文本: '{cleaned_text}'")
        
        normalized_text = TextUtils.normalize_text(cleaned_text)
        logger.debug(f"标准化后的文本: '{normalized_text}'")
        
        # 提取中文词汇
        chinese_words = TextUtils.extract_chinese_words(normalized_text)
        logger.debug(f"中文词汇: {chinese_words}")
        
        # 提取英文词汇（修复英文关键词提取问题）
        # 按空白字符分割并过滤非字母标记
        english_words = []
        for word in normalized_text.split():
            # 只保留字母字符并确保最小长度
            alphabetic_word = ''.join([c for c in word if c.isalpha()])
            if len(alphabetic_word) >= 2:  # Minimum length for English words
                english_words.append(alphabetic_word)
        
        logger.debug(f"英文词汇: {english_words}")
        
        # 组合中文和英文关键词
        all_keywords = chinese_words + english_words
        
        # 按长度过滤并去重
        keywords = TextUtils.filter_by_length(all_keywords, min_length=2)
        keywords = TextUtils.remove_duplicates(keywords)
        
        # 确保所有关键词都是小写
        keywords = [keyword.lower() for keyword in keywords]
        logger.debug(f"最终关键词: {keywords}")
        
        return keywords
    
    def get_all_qa_data(self) -> List[QAData]:
        """
        获取所有已加载的QA数据。
        
        返回:
            List[QAData]: 所有QA数据对象的列表
            
        异常:
            RuntimeError: 如果数据尚未加载
        """
        if not self.is_loaded:
            raise RuntimeError("数据尚未加载，请先调用load_data()")
            
        return self.qa_data_list.copy()
    
    def get_qa_by_id(self, qa_id: str) -> Optional[QAData]:
        """
        根据ID获取QA数据。
        
        参数:
            qa_id: 要检索的QA数据的ID
            
        返回:
            Optional[QAData]: 如果找到则返回QA数据对象，否则返回None
            
        异常:
            RuntimeError: 如果数据尚未加载
        """
        if not self.is_loaded:
            raise RuntimeError("数据尚未加载，请先调用load_data()")
            
        return self.id_to_qa.get(qa_id)
    
    def get_qa_count(self) -> int:
        """
        获取已加载的QA记录数量。
        
        返回:
            int: QA记录数量
            
        异常:
            RuntimeError: 如果数据尚未加载
        """
        if not self.is_loaded:
            raise RuntimeError("数据尚未加载，请先调用load_data()")
            
        return len(self.qa_data_list)
    
    def get_all_questions(self) -> List[str]:
        """
        获取所有问题文本。
        
        返回:
            List[str]: 问题文本列表
            
        异常:
            RuntimeError: 如果数据尚未加载
        """
        if not self.is_loaded:
            raise RuntimeError("数据尚未加载，请先调用load_data()")
            
        return [qa.question for qa in self.qa_data_list]
    
    def get_all_answers(self) -> List[str]:
        """
        获取所有答案文本。
        
        返回:
            List[str]: 答案文本列表
            
        异常:
            RuntimeError: 如果数据尚未加载
        """
        if not self.is_loaded:
            raise RuntimeError("数据尚未加载，请先调用load_data()")
            
        return [qa.answer for qa in self.qa_data_list]
    
    def get_all_keywords(self) -> List[str]:
        """
        获取所有唯一关键词。
        
        返回:
            List[str]: 唯一关键词列表
            
        异常:
            RuntimeError: 如果数据尚未加载
        """
        if not self.is_loaded:
            raise RuntimeError("数据尚未加载，请先调用load_data()")
            
        all_keywords = set()
        for qa in self.qa_data_list:
            all_keywords.update(qa.keywords)
            
        return list(all_keywords)
    
    def search_by_keyword(self, keyword: str, field: str = 'all') -> List[QAData]:
        """
        搜索包含特定关键词的QA数据。
        
        参数:
            keyword: 要搜索的关键词
            field: 搜索字段（'question'、'answer'或'all'）
            
        返回:
            List[QAData]: 匹配的QA数据对象列表
            
        异常:
            RuntimeError: 如果数据尚未加载
            ValueError: 如果字段不被识别
        """
        if not self.is_loaded:
            raise RuntimeError("数据尚未加载，请先调用load_data()")
            
        keyword = keyword.lower()
        results = []
        
        for qa in self.qa_data_list:
            if qa.has_keyword(keyword, field):
                results.append(qa)
                
        return results
    
    def search_by_keywords(self, keywords: List[str], field: str = 'all', match_all: bool = False) -> List[QAData]:
        """
        搜索包含多个关键词的QA数据。
        
        参数:
            keywords: 要搜索的关键词列表
            field: 搜索字段（'question'、'answer'或'all'）
            match_all: 如果为True，要求所有关键词都匹配；如果为False，任意关键词匹配即可
            
        返回:
            List[QAData]: 匹配的QA数据对象列表
            
        异常:
            RuntimeError: 如果数据尚未加载
            ValueError: 如果字段不被识别
        """
        if not self.is_loaded:
            raise RuntimeError("数据尚未加载，请先调用load_data()")
            
        if not keywords:
            return []
            
        results = []
        
        for qa in self.qa_data_list:
            keyword_matches = [qa.has_keyword(keyword.lower(), field) for keyword in keywords]
            
            if match_all:
                # 要求所有关键词都匹配
                if all(keyword_matches):
                    results.append(qa)
            else:
                # 任意关键词匹配即可
                if any(keyword_matches):
                    results.append(qa)
                    
        return results
    
    def get_keywords_by_field(self, field: str = 'all') -> List[str]:
        """
        从特定字段获取所有唯一关键词。
        
        参数:
            field: 获取关键词的字段（'question'、'answer'或'all'）
            
        返回:
            List[str]: 唯一关键词列表
            
        异常:
            RuntimeError: 如果数据尚未加载
            ValueError: 如果字段不被识别
        """
        if not self.is_loaded:
            raise RuntimeError("数据尚未加载，请先调用load_data()")
            
        all_keywords = set()
        for qa in self.qa_data_list:
            all_keywords.update(qa.get_keywords_by_field(field))
            
        return list(all_keywords)
    
    def reload_data(self) -> None:
        """
        从文件重新加载数据。
        
        当数据文件已更新时这很有用。
        """
        logger.info("重新加载数据")
        self.load_data()
    
    def is_data_loaded(self) -> bool:
        """
        检查数据是否已加载。
        
        返回:
            bool: 如果数据已加载则返回True，否则返回False
        """
        return self.is_loaded