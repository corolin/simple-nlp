"""
Data models for the NLP search system.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class QAData(BaseModel):
    """
    Question-Answer data model for storing QA pairs.
    
    Attributes:
        id: Unique identifier for the QA pair
        question: Question text
        answer: Answer text
        keywords: List of keywords extracted from the question
        answer_keywords: List of keywords extracted from the answer
        all_keywords: Combined list of keywords from both question and answer
        create_time: Creation timestamp
    """
    id: str = Field(..., description="Unique identifier for the QA pair")
    question: str = Field(..., description="Question text")
    answer: str = Field(..., description="Answer text")
    keywords: List[str] = Field(default_factory=list, description="List of keywords extracted from the question")
    answer_keywords: List[str] = Field(default_factory=list, description="List of keywords extracted from the answer")
    all_keywords: List[str] = Field(default_factory=list, description="Combined list of keywords from both question and answer")
    create_time: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")
    
    class Config:
        """Pydantic模型配置。"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        
    def update_all_keywords(self) -> None:
        """
        Update the all_keywords field by combining question and answer keywords.
        
        This method merges keywords from both question and answer fields,
        removes duplicates, and maintains order of first occurrence.
        """
        # Combine keywords from both fields
        combined_keywords = self.keywords + self.answer_keywords
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in combined_keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        self.all_keywords = unique_keywords
    
    def get_keywords_by_field(self, field: str) -> List[str]:
        """
        Get keywords for a specific field.
        
        Args:
            field: Field name ('question', 'answer', or 'all')
            
        Returns:
            List[str]: Keywords for the specified field
            
        Raises:
            ValueError: If field is not recognized
        """
        if field == 'question':
            return self.keywords
        elif field == 'answer':
            return self.answer_keywords
        elif field == 'all':
            return self.all_keywords
        else:
            raise ValueError(f"Unknown field: {field}. Valid fields are: question, answer, all")
    
    def has_keyword(self, keyword: str, field: str = 'all') -> bool:
        """
        Check if a keyword exists in the specified field.
        
        Args:
            keyword: Keyword to search for
            field: Field name ('question', 'answer', or 'all')
            
        Returns:
            bool: True if keyword exists, False otherwise
        """
        keywords = self.get_keywords_by_field(field)
        keyword_lower = keyword.lower()
        
        # DEBUG: 添加日志记录
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"检查关键词 '{keyword}' (小写: '{keyword_lower}') 是否在字段 '{field}' 的关键词中: {keywords}")
        
        # 由于关键词在提取时已经统一转换为小写，这里只需要将查询词转换为小写
        result = keyword_lower in keywords
        logger.debug(f"关键词匹配结果: {result}")
        
        return result
    
    def __str__(self) -> str:
        """QAData的字符串表示。"""
        return f"QAData(id={self.id}, question={self.question[:50]}..., answer={self.answer[:50]}...)"
    
    def __repr__(self) -> str:
        """QAData的表示。"""
        return self.__str__()