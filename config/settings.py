"""
Configuration Settings for OAI_BOT_V.1
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings"""
    
    # App Configuration
    app_name: str = "OAI_BOT_V.1"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # AI Service Configuration
    embedding_api_url: str = Field(
        default="http://209.15.123.47:11434/api/embeddings",
        env="EMBEDDING_API_URL"
    )
    embedding_model: str = Field(
        default="nomic-embed-text:latest",
        env="EMBEDDING_MODEL"
    )
    chat_api_url: str = Field(
        default="http://209.15.123.47:11434/api/generate",
        env="CHAT_API_URL"
    )
    chat_model: str = Field(
        default="Qwen3:14b",
        env="CHAT_MODEL"
    )
    
    # Thai OCR Models
    thai_llm_model: str = Field(
        default="scb10x/llama3.1-typhoon2-8b-instruct:latest",
        env="THAI_LLM_MODEL"
    )
    thai_ocr_model: str = Field(
        default="scb10x/typhoon-ocr-7b:latest",
        env="THAI_OCR_MODEL"
    )
    
    # Qdrant Configuration
    qdrant_url: str = Field(
        default="https://b01ce415-b0a5-4ff3-b67a-54efe7f8d22b.europe-west3-0.gcp.cloud.qdrant.io:6333",
        env="QDRANT_URL"
    )
    qdrant_api_key: str = Field(
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.gkxs8PPOwSjROysezAReXHcSIKVVAo7KOUohTPL4ZDA",
        env="QDRANT_API_KEY"
    )
    qdrant_collection_name: str = Field(
        default="oai_bot_documents",
        env="QDRANT_COLLECTION_NAME"
    )
    
    # File Upload Configuration
    upload_folder: str = Field(default="uploads", env="UPLOAD_FOLDER")
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    allowed_extensions: list = Field(
        default=['.pdf', '.doc', '.docx', '.txt'],
        env="ALLOWED_EXTENSIONS"
    )
    
    # Line Notification (configurable in settings)
    line_notify_token: Optional[str] = Field(default=None, env="LINE_NOTIFY_TOKEN")
    
    # System Prompt (configurable in settings)
    default_system_prompt: str = Field(
        default="""คุณเป็น AI Assistant ที่ช่วยวิเคราะห์เอกสารสำหรับงาน บลนป. (บริษัทลิสซิ่งนครหลวง จำกัด)
        
        หน้าที่ของคุณ:
        1. ตอบคำถามเกี่ยวกับเอกสารที่อัพโหลดมา
        2. สรุปเนื้อหาสำคัญของเอกสาร
        3. ค้นหาข้อมูลตามที่ผู้ใช้ต้องการ
        4. ให้คำแนะนำการทำงานที่เกี่ยวข้องกับเอกสาร
        
        กรุณาตอบเป็นภาษาไทยที่สุภาพและเป็นมิตร มีการอ้างอิงข้อมูลจากเอกสารที่เกี่ยวข้อง""",
        env="DEFAULT_SYSTEM_PROMPT"
    )
    
    # Admin Helper Prompt
    admin_system_prompt: str = Field(
        default="""คุณเป็น Admin Assistant สำหรับระบบ OAI_BOT_V.1
        
        หน้าที่ของคุณ:
        1. ช่วยผู้ใช้เข้าใจการทำงานของระบบ
        2. แนะนำขั้นตอนการใช้งาน
        3. ช่วยออกแบบคำถามที่เหมาะสม
        4. แก้ไขปัญหาการใช้งาน
        5. อธิบายฟีเจอร์ต่างๆ ของระบบ
        
        ตอบเป็นภาษาไทยที่เข้าใจง่าย และให้ขั้นตอนที่ชัดเจน""",
        env="ADMIN_SYSTEM_PROMPT"
    )
    
    # Embedding Configuration
    embedding_dimension: int = Field(default=768, env="EMBEDDING_DIMENSION")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # RAG Configuration
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_upload_path(self, filename: str) -> str:
        """Get full upload path for a file"""
        return os.path.join(self.upload_folder, filename)
    
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return any(filename.lower().endswith(ext) for ext in self.allowed_extensions)
