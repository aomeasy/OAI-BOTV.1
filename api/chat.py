"""
Chat API endpoints for OAI_BOT_V.1
"""

import logging
from flask import Blueprint, request, jsonify
from datetime import datetime

from core.rag_engine import RAGEngine
from core.ai_service import AIService
from config.settings import Settings

logger = logging.getLogger(__name__)

# Create blueprint
chat_bp = Blueprint('chat', __name__)

# Initialize services
settings = Settings()
rag_engine = RAGEngine(settings)
ai_service = AIService(settings)

# Simple in-memory chat storage (in production, use proper database)
chat_sessions = {}

@chat_bp.route('/message', methods=['POST'])
async def send_message():
    """Send a message and get AI response"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "Message is required"
            }), 400
        
        message = data['message']
        session_id = data.get('session_id', 'default')
        use_rag = data.get('use_rag', True)
        system_prompt = data.get('system_prompt')
        
        # Get or create chat session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "messages": [],
                "created_at": datetime.now().isoformat()
            }
        
        session = chat_sessions[session_id]
        
        # Add user message to session
        user_message = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        session["messages"].append(user_message)
        
        # Generate response
        if use_rag:
            # Use RAG for context-aware response
            response_result = await rag_engine.chat_with_context(
                messages=session["messages"],
                system_prompt=system_prompt,
                auto_retrieve=True
            )
        else:
            # Use direct AI without RAG
            response_result = await ai_service.generate_chat_response(
                messages=session["messages"],
                system_prompt=system_prompt
            )
            
            # Format response to match RAG structure
            if response_result["success"]:
                response_result.update({
                    "sources": [],
                    "context_used": False
                })
        
        if not response_result["success"]:
            return jsonify(response_result), 500
        
        # Add assistant message to session
        assistant_message = {
            "role": "assistant",
            "content": response_result["response"],
            "timestamp": datetime.now().isoformat(),
            "sources": response_result.get("sources", []),
            "context_used": response_result.get("context_used", False)
        }
        session["messages"].append(assistant_message)
        
        # Limit session history (keep last 20 messages)
        if len(session["messages"]) > 20:
            session["messages"] = session["messages"][-20:]
        
        return jsonify({
            "success": True,
            "response": response_result["response"],
            "session_id": session_id,
            "sources": response_result.get("sources", []),
            "context_used": response_result.get("context_used", False),
            "model_used": response_result.get("model_used"),
            "timestamp": assistant_message["timestamp"]
        })
        
    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@chat_bp.route('/query', methods=['POST'])
async def query_documents():
    """Query documents directly (single question without session)"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Query is required"
            }), 400
        
        query = data['query']
        system_prompt = data.get('system_prompt')
        max_results = data.get('max_results', 5)
        similarity_threshold = data.get('similarity_threshold', 0.7)
        
        # Query using RAG engine
        result = await rag_engine.query_documents(
            query=query,
            system_prompt=system_prompt,
            max_results=max_results,
            similarity_threshold=similarity_threshold
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in query_documents: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@chat_bp.route('/session/<session_id>', methods=['GET'])
def get_chat_session(session_id):
    """Get chat session history"""
    try:
        if session_id not in chat_sessions:
            return jsonify({
                "success": False,
                "error": "Session not found"
            }), 404
        
        session = chat_sessions[session_id]
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "messages": session["messages"],
            "created_at": session["created_at"],
            "message_count": len(session["messages"])
        })
        
    except Exception as e:
        logger.error(f"Error in get_chat_session: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@chat_bp.route('/session/<session_id>', methods=['DELETE'])
def delete_chat_session(session_id):
    """Delete chat session"""
    try:
        if session_id in chat_sessions:
            del chat_sessions[session_id]
            return jsonify({
                "success": True,
                "message": "Session deleted successfully"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Session not found"
            }), 404
        
    except Exception as e:
        logger.error(f"Error in delete_chat_session: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@chat_bp.route('/sessions', methods=['GET'])
def list_chat_sessions():
    """List all chat sessions"""
    try:
        sessions_info = []
        
        for session_id, session in chat_sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "created_at": session["created_at"],
                "message_count": len(session["messages"]),
                "last_message": session["messages"][-1]["timestamp"] if session["messages"] else None
            })
        
        # Sort by last activity
        sessions_info.sort(key=lambda x: x["last_message"] or x["created_at"], reverse=True)
        
        return jsonify({
            "success": True,
            "sessions": sessions_info,
            "total_sessions": len(sessions_info)
        })
        
    except Exception as e:
        logger.error(f"Error in list_chat_sessions: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@chat_bp.route('/count', methods=['GET'])
def get_chat_count():
    """Get total chat count"""
    try:
        total_messages = sum(len(session["messages"]) for session in chat_sessions.values())
        
        return jsonify({
            "success": True,
            "count": total_messages,
            "sessions": len(chat_sessions)
        })
        
    except Exception as e:
        logger.error(f"Error in get_chat_count: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "count": 0
        }), 500

@chat_bp.route('/models', methods=['GET'])
def get_available_models():
    """Get available AI models"""
    try:
        return jsonify({
            "success": True,
            "models": {
                "chat_model": settings.chat_model,
                "embedding_model": settings.embedding_model,
                "thai_llm_model": settings.thai_llm_model,
                "thai_ocr_model": settings.thai_ocr_model
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_available_models: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@chat_bp.route('/clear-all', methods=['POST'])
def clear_all_sessions():
    """Clear all chat sessions"""
    try:
        global chat_sessions
        session_count = len(chat_sessions)
        chat_sessions = {}
        
        return jsonify({
            "success": True,
            "message": f"Cleared {session_count} chat sessions"
        })
        
    except Exception as e:
        logger.error(f"Error in clear_all_sessions: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@chat_bp.route('/suggest-questions', methods=['POST'])
async def suggest_questions():
    """Suggest questions based on available documents"""
    try:
        data = request.get_json()
        document_context = data.get('document_context', '')
        
        # Create a prompt to generate suggested questions
        suggestion_prompt = """
        โปรดแนะนำ 5 คำถามที่น่าสนใจที่ผู้ใช้สามารถถามเกี่ยวกับเอกสารที่มีอยู่ในระบบ คำถามควรจะ:
        1. เป็นประโยชน์และเกี่ยวข้องกับงาน
        2. ช่วยให้ผู้ใช้เข้าใจเนื้อหาเอกสารมากขึ้น
        3. สามารถตอบได้จากข้อมูลในเอกสาร
        
        โปรดตอบเป็นรายการคำถาม 5 ข้อ แต่ละข้อในบรรทัดใหม่ เริ่มด้วยหมายเลข
        """
        
        if document_context:
            suggestion_prompt += f"\n\nบริบทเอกสาร: {document_context[:1000]}"
        
        # Generate suggestions using AI
        messages = [{"role": "user", "content": suggestion_prompt}]
        response_result = await ai_service.generate_chat_response(
            messages=messages,
            system_prompt="คุณเป็นผู้เชี่ยวชาญในการแนะนำคำถามที่มีประโยชน์"
        )
        
        if not response_result["success"]:
            return jsonify({
                "success": False,
                "error": "Failed to generate suggestions"
            }), 500
        
        # Parse suggestions (simple parsing)
        suggestions_text = response_result["response"]
        suggestions = []
        
        for line in suggestions_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering and clean up
                clean_question = line.split('.', 1)[-1].strip()
                if clean_question:
                    suggestions.append(clean_question)
        
        return jsonify({
            "success": True,
            "suggestions": suggestions[:5],  # Limit to 5
            "raw_response": suggestions_text
        })
        
    except Exception as e:
        logger.error(f"Error in suggest_questions: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500
