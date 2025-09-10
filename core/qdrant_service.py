"""
Qdrant Vector Database Service for OAI_BOT_V.1
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams, Distance, PointStruct, 
        Filter, FieldCondition, Match, SearchRequest
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not available. Install with: pip install qdrant-client")

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self, settings):
        self.settings = settings
        self.client = None
        self.collection_name = settings.qdrant_collection_name
        
        if QDRANT_AVAILABLE:
            try:
                self.client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    timeout=60
                )
                self._ensure_collection_exists()
                logger.info("Qdrant client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant client: {str(e)}")
                self.client = None
        else:
            logger.error("Qdrant client not available")
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if not"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.settings.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
    
    async def add_documents(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add documents with embeddings to Qdrant"""
        try:
            if not self.client:
                return {"success": False, "error": "Qdrant client not available"}
            
            if len(texts) != len(embeddings) or len(texts) != len(metadata):
                return {"success": False, "error": "Mismatched input lengths"}
            
            points = []
            point_ids = []
            
            for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                # Prepare payload
                payload = {
                    "text": text,
                    "document_id": meta.get("document_id"),
                    "filename": meta.get("filename"),
                    "chunk_index": meta.get("chunk_index", i),
                    "created_at": datetime.now().isoformat(),
                    **meta
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
            
            # Upload points to Qdrant
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(points)} points to Qdrant")
            
            return {
                "success": True,
                "point_ids": point_ids,
                "count": len(points),
                "operation_id": result.operation_id if hasattr(result, 'operation_id') else None
            }
            
        except Exception as e:
            logger.error(f"Error adding documents to Qdrant: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        limit: int = 5,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar documents"""
        try:
            if not self.client:
                return {"success": False, "error": "Qdrant client not available"}
            
            # Prepare filter if provided
            query_filter = None
            if filter_conditions:
                conditions = []
                for field, value in filter_conditions.items():
                    conditions.append(FieldCondition(
                        key=field,
                        match=Match(value=value)
                    ))
                query_filter = Filter(must=conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
            
            # Format results
            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "metadata": {
                        k: v for k, v in hit.payload.items() 
                        if k != "text"
                    }
                })
            
            logger.info(f"Found {len(results)} similar documents")
            
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_document_chunks(self, document_id: str) -> Dict[str, Any]:
        """Get all chunks for a specific document"""
        try:
            if not self.client:
                return {"success": False, "error": "Qdrant client not available"}
            
            # Search with filter for specific document
            filter_condition = Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=Match(value=document_id)
                )]
            )
            
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1000  # Adjust based on expected document size
            )
            
            chunks = []
            for point in search_results[0]:  # scroll returns (points, next_page_offset)
                chunks.append({
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "metadata": point.payload
                })
            
            # Sort by chunk_index
            chunks.sort(key=lambda x: x.get("chunk_index", 0))
            
            return {
                "success": True,
                "chunks": chunks,
                "count": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete all chunks for a specific document"""
        try:
            if not self.client:
                return {"success": False, "error": "Qdrant client not available"}
            
            # Delete points with specific document_id
            filter_condition = Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=Match(value=document_id)
                )]
            )
            
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition
            )
            
            logger.info(f"Deleted document {document_id} from Qdrant")
            
            return {
                "success": True,
                "operation_id": result.operation_id if hasattr(result, 'operation_id') else None
            }
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            if not self.client:
                return {"success": False, "error": "Qdrant client not available"}
            
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "success": True,
                "info": {
                    "vectors_count": collection_info.vectors_count,
                    "points_count": collection_info.points_count,
                    "status": collection_info.status.value,
                    "optimizer_status": collection_info.optimizer_status.value if collection_info.optimizer_status else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def health_check(self) -> bool:
        """Check if Qdrant service is healthy"""
        try:
            if not self.client:
                return False
            
            # Try to get collection info
            self.client.get_collection(self.collection_name)
            return True
            
        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            return False
    
    async def count_documents(self) -> int:
        """Count total number of documents (unique document_ids)"""
        try:
            if not self.client:
                return 0
            
            # This is a simplified count - in production you might want to
            # implement a more sophisticated document counting mechanism
            collection_info = await self.get_collection_info()
            if collection_info["success"]:
                return collection_info["info"]["points_count"]
            return 0
            
        except Exception as e:
            logger.error(f"Error counting documents: {str(e)}")
            return 0
