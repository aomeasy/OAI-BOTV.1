"""
Document API endpoints for OAI_BOT_V.1
"""

import os
import logging
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from core.document_processor import DocumentProcessor
from core.rag_engine import RAGEngine
from config.settings import Settings

logger = logging.getLogger(__name__)

# Create blueprint
documents_bp = Blueprint('documents', __name__)

# Initialize services
settings = Settings()
doc_processor = DocumentProcessor(settings)
rag_engine = RAGEngine(settings)

@documents_bp.route('/upload', methods=['POST'])
async def upload_document():
    """Upload and process a document"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Secure filename
        filename = secure_filename(file.filename)
        
        # Validate file
        file_data = file.read()
        validation = doc_processor.validate_file(filename, len(file_data))
        
        if not validation["valid"]:
            return jsonify({
                "success": False,
                "errors": validation["errors"]
            }), 400
        
        # Save file
        file_path = doc_processor.save_uploaded_file(file_data, filename)
        
        # Process document
        process_result = await doc_processor.process_document(file_path, filename)
        
        if not process_result["success"]:
            # Clean up file if processing failed
            doc_processor.delete_file(file_path)
            return jsonify(process_result), 400
        
        # Store in RAG system
        rag_result = await rag_engine.process_and_store_document(
            text_chunks=process_result["chunks"],
            metadata=process_result["metadata"]
        )
        
        if not rag_result["success"]:
            # Clean up file if RAG processing failed
            doc_processor.delete_file(file_path)
            return jsonify({
                "success": False,
                "error": f"Failed to store in RAG system: {rag_result['error']}"
            }), 500
        
        # Clean up temporary file
        doc_processor.delete_file(file_path)
        
        return jsonify({
            "success": True,
            "message": "Document uploaded and processed successfully",
            "document_id": process_result["document_id"],
            "filename": filename,
            "chunks_processed": rag_result["chunks_processed"],
            "metadata": process_result["metadata"]
        })
        
    except Exception as e:
        logger.error(f"Error in upload_document: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@documents_bp.route('/list', methods=['GET'])
async def list_documents():
    """List all documents"""
    try:
        # Get collection info
        collection_info = await rag_engine.qdrant_service.get_collection_info()
        
        if not collection_info["success"]:
            return jsonify({
                "success": False,
                "error": "Failed to get documents list"
            }), 500
        
        # For now, return basic info
        # In a production system, you'd want to maintain a separate documents table
        return jsonify({
            "success": True,
            "total_points": collection_info["info"]["points_count"],
            "message": "Use /search endpoint to find specific documents"
        })
        
    except Exception as e:
        logger.error(f"Error in list_documents: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@documents_bp.route('/search', methods=['POST'])
async def search_documents():
    """Search documents"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Query is required"
            }), 400
        
        query = data['query']
        max_results = data.get('max_results', 5)
        similarity_threshold = data.get('similarity_threshold', 0.7)
        
        # Search using RAG engine
        search_result = await rag_engine.query_documents(
            query=query,
            max_results=max_results,
            similarity_threshold=similarity_threshold
        )
        
        return jsonify(search_result)
        
    except Exception as e:
        logger.error(f"Error in search_documents: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500
