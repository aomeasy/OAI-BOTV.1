"""
RAG (Retrieval-Augmented Generation) Engine for OAI_BOT_V.1
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .ai_service import AIService
from .qdrant_service import QdrantService

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, settings):
        self.settings = settings
        self.ai_service = AIService(settings)
        self.qdrant_service = QdrantService(settings)
        
    async def process_and_store_document(
        self, 
        text_chunks: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process document chunks and store in vector database"""
        try:
            # Extract text from chunks
            texts = [chunk["text"] for chunk in text_chunks]
            
            # Generate embeddings
            embedding_result = await self.ai_service.generate_embeddings(texts)
            
            if not embedding_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to generate embeddings: {embedding_result['error']}"
                }
            
            embeddings = embedding_result["embeddings"]
            
            # Prepare metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(text_chunks):
                chunk_meta = {
                    **metadata,
                    "chunk_index": chunk["chunk_index"],
                    "character_count": chunk["character_count"],
                    "embedding_model": self.settings.embedding_model
                }
                chunk_metadata.append(chunk_meta)
            
            # Store in Qdrant
            store_result = await self.qdrant_service.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadata=chunk_metadata
            )
            
            if not store_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to store in vector database: {store_result['error']}"
                }
            
            logger.info(f"Successfully processed and stored document: {metadata.get('filename')}")
            
            return {
                "success": True,
                "document_id": metadata["document_id"],
                "chunks_processed": len(text_chunks),
                "embeddings_generated": len(embeddings),
                "points_stored": store_result["count"]
            }
            
        except Exception as e:
            logger.error(f"Error in process_and_store_document: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def query_documents(
        self, 
        query: str, 
        system_prompt: Optional[str] = None,
        max_results: int = 5,
        similarity_threshold: float = 0.7,
        document_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query documents using RAG"""
        try:
            # Generate embedding for query
            query_embedding_result = await self.ai_service.generate_embeddings([query])
            
            if not query_embedding_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to generate query embedding: {query_embedding_result['error']}"
                }
            
            query_embedding = query_embedding_result["embeddings"][0]
            
            # Search similar documents
            search_result = await self.qdrant_service.search_similar(
                query_embedding=query_embedding,
                limit=max_results,
                score_threshold=similarity_threshold,
                filter_conditions=document_filter
            )
            
            if not search_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to search documents: {search_result['error']}"
                }
            
            # Prepare context from retrieved documents
            context_chunks = search_result["results"]
            context_text = self._build_context_from_chunks(context_chunks)
            
            # Generate response using AI
            messages = [{"role": "user", "content": query}]
            
            # Enhanced system prompt with context
            enhanced_system_prompt = self._build_enhanced_system_prompt(
                system_prompt or self.settings.default_system_prompt,
                context_text,
                query
            )
            
            ai_response = await self.ai_service.generate_chat_response(
                messages=messages,
                system_prompt=enhanced_system_prompt
            )
            
            if not ai_response["success"]:
                return {
                    "success": False,
                    "error": f"Failed to generate AI response: {ai_response['error']}"
                }
            
            # Prepare sources information
            sources = self._extract_sources_info(context_chunks)
            
            return {
                "success": True,
                "response": ai_response["response"],
                "query": query,
                "sources": sources,
                "context_chunks": len(context_chunks),
                "model_used": ai_response["model"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in query_documents: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def chat_with_context(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        document_context: Optional[str] = None,
        auto_retrieve: bool = True
    ) -> Dict[str, Any]:
        """Chat with automatic context retrieval"""
        try:
            context_text = document_context or ""
            sources = []
            
            # Get the latest user message for context retrieval
            latest_message = messages[-1]["content"] if messages else ""
            
            # Auto-retrieve relevant context if enabled
            if auto_retrieve and latest_message:
                retrieval_result = await self._retrieve_relevant_context(latest_message)
                if retrieval_result["success"]:
                    context_text = retrieval_result["context"]
                    sources = retrieval_result["sources"]
            
            # Build enhanced system prompt
            enhanced_system_prompt = self._build_enhanced_system_prompt(
                system_prompt or self.settings.default_system_prompt,
                context_text,
                latest_message
            )
            
            # Generate AI response
            ai_response = await self.ai_service.generate_chat_response(
                messages=messages,
                system_prompt=enhanced_system_prompt
            )
            
            if not ai_response["success"]:
                return {
                    "success": False,
                    "error": f"Failed to generate response: {ai_response['error']}"
                }
            
            return {
                "success": True,
                "response": ai_response["response"],
                "sources": sources,
                "context_used": len(context_text) > 0,
                "model_used": ai_response["model"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in chat_with_context: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _retrieve_relevant_context(self, query: str, max_chunks: int = 3) -> Dict[str, Any]:
        """Retrieve relevant context for a query"""
        try:
            # Generate embedding for query
            query_embedding_result = await self.ai_service.generate_embeddings([query])
            
            if not query_embedding_result["success"]:
                return {"success": False, "context": "", "sources": []}
            
            query_embedding = query_embedding_result["embeddings"][0]
            
            # Search similar documents
            search_result = await self.qdrant_service.search_similar(
                query_embedding=query_embedding,
                limit=max_chunks,
                score_threshold=self.settings.similarity_threshold
            )
            
            if not search_result["success"] or not search_result["results"]:
                return {"success": True, "context": "", "sources": []}
            
            # Build context and sources
            context_text = self._build_context_from_chunks(search_result["results"])
            sources = self._extract_sources_info(search_result["results"])
            
            return {
                "success": True,
                "context": context_text,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return {"success": False, "context": "", "sources": []}
    
    def _build_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context text from retrieved chunks"""
        if not chunks:
            return ""
        
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "").strip()
            if chunk_text:
                metadata = chunk.get("metadata", {})
                filename = metadata.get("filename", "Unknown")
                chunk_index = metadata.get("chunk_index", i)
                
                context_parts.append(f"[เอกสาร: {filename}, ส่วนที่ {chunk_index + 1}]\n{chunk_text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_sources_info(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from chunks"""
        sources = []
        seen_docs = set()
        
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("document_id")
            
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                sources.append({
                    "document_id": doc_id,
                    "filename": metadata.get("filename", "Unknown"),
                    "relevance_score": chunk.get("score", 0.0),
                    "chunk_count": 1  # Will be updated if more chunks from same doc
                })
            elif doc_id in seen_docs:
                # Update chunk count for existing document
                for source in sources:
                    if source["document_id"] == doc_id:
                        source["chunk_count"] += 1
                        # Update relevance score to highest
                        if chunk.get("score", 0) > source["relevance_score"]:
                            source["relevance_score"] = chunk.get("score", 0)
                        break
        
        return sources
    
    def _build_enhanced_system_prompt(
        self, 
        base_prompt: str, 
        context: str, 
        query: str
    ) -> str:
        """Build enhanced system prompt with context"""
        if not context:
            return base_prompt
        
        enhanced_prompt = f"""{base_prompt}

ข้อมูลที่เกี่ยวข้องจากเอกสาร:
{context}

ใช้ข้อมูลจากเอกสารข้างต้นในการตอบคำถาม หากไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร ให้แจ้งให้ทราบและตอบตามความรู้ทั่วไป

คำถาม: {query}
"""
        
        return enhanced_prompt
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete document from vector database"""
        try:
            result = await self.qdrant_service.delete_document(document_id)
            
            if result["success"]:
                logger.info(f"Document {document_id} deleted from RAG system")
            
            return result
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """Get summary of a specific document"""
        try:
            # Get all chunks for the document
            chunks_result = await self.qdrant_service.get_document_chunks(document_id)
            
            if not chunks_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to get document chunks: {chunks_result['error']}"
                }
            
            chunks = chunks_result["chunks"]
            if not chunks:
                return {
                    "success": False,
                    "error": "No chunks found for document"
                }
            
            # Combine chunks to get full text
            full_text = " ".join([chunk["text"] for chunk in chunks])
            
            # Generate summary using AI
            summary_prompt = f"กรุณาสรุปเนื้อหาของเอกสารนี้ให้อ่านง่ายและเข้าใจง่าย:\n\n{full_text[:4000]}"  # Limit to 4000 chars
            
            messages = [{"role": "user", "content": summary_prompt}]
            ai_response = await self.ai_service.generate_chat_response(
                messages=messages,
                system_prompt="คุณเป็นผู้เชี่ยวชาญในการสรุปเอกสาร สรุปให้กระชับแต่ครอบคลุมประเด็นสำคัญ"
            )
            
            if not ai_response["success"]:
                return {
                    "success": False,
                    "error": f"Failed to generate summary: {ai_response['error']}"
                }
            
            # Get document metadata
            metadata = chunks[0].get("metadata", {}) if chunks else {}
            
            return {
                "success": True,
                "document_id": document_id,
                "summary": ai_response["response"],
                "chunk_count": len(chunks),
                "total_characters": len(full_text),
                "filename": metadata.get("filename", "Unknown"),
                "processed_at": metadata.get("created_at")
            }
            
        except Exception as e:
            logger.error(f"Error generating document summary: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def health_check(self) -> bool:
        """Check if RAG engine is healthy"""
        try:
            ai_healthy = self.ai_service.health_check()
            qdrant_healthy = self.qdrant_service.health_check()
            
            return ai_healthy and qdrant_healthy
            
        except Exception as e:
            logger.error(f"RAG engine health check failed: {str(e)}")
            return False
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            stats = {
                "rag_engine_healthy": self.health_check(),
                "ai_service_healthy": self.ai_service.health_check(),
                "qdrant_healthy": self.qdrant_service.health_check()
            }
            
            # Get collection info
            collection_info = await self.qdrant_service.get_collection_info()
            if collection_info["success"]:
                stats.update({
                    "total_points": collection_info["info"]["points_count"],
                    "vectors_count": collection_info["info"]["vectors_count"],
                    "collection_status": collection_info["info"]["status"]
                })
            
            return {
                "success": True,
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
