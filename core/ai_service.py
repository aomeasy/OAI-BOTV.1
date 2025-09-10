"""
AI Service for OAI_BOT_V.1
Handles communication with AI models
"""

import asyncio
import httpx
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self, settings):
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=120.0)
        
    async def generate_chat_response(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate chat response using specified model"""
        try:
            # Use custom system prompt or default
            if system_prompt is None:
                system_prompt = self.settings.default_system_prompt
            
            # Use custom model or default
            if model is None:
                model = self.settings.chat_model
            
            # Prepare messages with system prompt
            formatted_messages = [{"role": "system", "content": system_prompt}]
            formatted_messages.extend(messages)
            
            # Create prompt for Ollama-style API
            prompt = self._format_messages_for_ollama(formatted_messages)
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1
                }
            }
            
            logger.info(f"Sending chat request to {self.settings.chat_api_url}")
            
            response = await self.client.post(
                self.settings.chat_api_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": model,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.error(f"Chat API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            logger.error(f"Error in generate_chat_response: {str(e)}")
            return {
                "success": False,
                "error": "Internal error",
                "details": str(e)
            }
    
    async def generate_thai_response(
        self, 
        prompt: str, 
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using Thai LLM model"""
        try:
            # Prepare prompt for Thai model
            if context:
                full_prompt = f"บริบท: {context}\n\nคำถาม: {prompt}\n\nคำตอบ:"
            else:
                full_prompt = prompt
            
            payload = {
                "model": self.settings.thai_llm_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = await self.client.post(
                self.settings.chat_api_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": self.settings.thai_llm_model
                }
            else:
                return {
                    "success": False,
                    "error": f"Thai LLM API Error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error in generate_thai_response: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_ocr(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """Process OCR using Thai OCR model"""
        try:
            # For now, this is a placeholder for OCR processing
            # In real implementation, you would send image data to OCR model
            
            payload = {
                "model": self.settings.thai_ocr_model,
                "prompt": "Please extract all Thai text from this image.",
                "stream": False
            }
            
            # Note: Actual OCR implementation would require image handling
            # This is a simplified version
            
            response = await self.client.post(
                self.settings.chat_api_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "extracted_text": result.get("response", ""),
                    "filename": filename,
                    "model": self.settings.thai_ocr_model
                }
            else:
                return {
                    "success": False,
                    "error": f"OCR API Error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error in process_ocr: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """Generate embeddings for texts"""
        try:
            all_embeddings = []
            
            for text in texts:
                payload = {
                    "model": self.settings.embedding_model,
                    "prompt": text
                }
                
                response = await self.client.post(
                    self.settings.embedding_api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding", [])
                    all_embeddings.append(embedding)
                else:
                    logger.error(f"Embedding API error: {response.status_code}")
                    return {
                        "success": False,
                        "error": f"Embedding API Error: {response.status_code}"
                    }
            
            return {
                "success": True,
                "embeddings": all_embeddings,
                "count": len(all_embeddings),
                "model": self.settings.embedding_model
            }
            
        except Exception as e:
            logger.error(f"Error in generate_embeddings: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _format_messages_for_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Ollama-style API"""
        formatted = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"Human: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        
        formatted += "Assistant: "
        return formatted
    
    def health_check(self) -> bool:
        """Check if AI service is healthy"""
        try:
            # Synchronous health check
            import requests
            response = requests.get(
                self.settings.chat_api_url.replace("/api/generate", "/"),
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            asyncio.create_task(self.close())
        except Exception:
            pass
