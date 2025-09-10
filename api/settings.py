"""
Settings API endpoints for OAI_BOT_V.1
"""

import logging
import json
import os
from flask import Blueprint, request, jsonify
from datetime import datetime

from config.settings import Settings

logger = logging.getLogger(__name__)

# Create blueprint
settings_bp = Blueprint('settings', __name__)

# Initialize settings
settings = Settings()

# Settings storage (in production, use proper database)
SETTINGS_FILE = "app_settings.json"

def load_app_settings():
    """Load application settings from file"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Default settings
            return {
                "system_prompt": settings.default_system_prompt,
                "admin_system_prompt": settings.admin_system_prompt,
                "line_notify_token": "",
                "similarity_threshold": settings.similarity_threshold,
                "top_k_results": settings.top_k_results,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error loading app settings: {str(e)}")
        return {}

def save_app_settings(settings_data):
    """Save application settings to file"""
    try:
        settings_data["updated_at"] = datetime.now().isoformat()
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving app settings: {str(e)}")
        return False

@settings_bp.route('/', methods=['GET'])
def get_settings():
    """Get current settings"""
    try:
        app_settings = load_app_settings()
        
        # Add current system info
        system_info = {
            "embedding_model": settings.embedding_model,
            "chat_model": settings.chat_model,
            "thai_llm_model": settings.thai_llm_model,
            "thai_ocr_model": settings.thai_ocr_model,
            "max_file_size_mb": settings.max_file_size / (1024 * 1024),
            "allowed_extensions": settings.allowed_extensions,
            "qdrant_collection": settings.qdrant_collection_name
        }
        
        return jsonify({
            "success": True,
            "settings": app_settings,
            "system_info": system_info
        })
        
    except Exception as e:
        logger.error(f"Error in get_settings: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@settings_bp.route('/', methods=['POST'])
def update_settings():
    """Update settings"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        # Load current settings
        current_settings = load_app_settings()
        
        # Update allowed fields
        allowed_fields = [
            'system_prompt',
            'admin_system_prompt', 
            'line_notify_token',
            'similarity_threshold',
            'top_k_results',
            'chunk_size',
            'chunk_overlap'
        ]
        
        updated_fields = []
        for field in allowed_fields:
            if field in data:
                current_settings[field] = data[field]
                updated_fields.append(field)
        
        # Validate settings
        validation_errors = validate_settings(current_settings)
        if validation_errors:
            return jsonify({
                "success": False,
                "errors": validation_errors
            }), 400
        
        # Save settings
        if save_app_settings(current_settings):
            return jsonify({
                "success": True,
                "message": "Settings updated successfully",
                "updated_fields": updated_fields,
                "settings": current_settings
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to save settings"
            }), 500
        
    except Exception as e:
        logger.error(f"Error in update_settings: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@settings_bp.route('/system-prompt', methods=['GET'])
def get_system_prompt():
    """Get current system prompt"""
    try:
        app_settings = load_app_settings()
        return jsonify({
            "success": True,
            "system_prompt": app_settings.get("system_prompt", settings.default_system_prompt),
            "admin_system_prompt": app_settings.get("admin_system_prompt", settings.admin_system_prompt)
        })
        
    except Exception as e:
        logger.error(f"Error in get_system_prompt: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@settings_bp.route('/system-prompt', methods=['POST'])
def update_system_prompt():
    """Update system prompt"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        app_settings = load_app_settings()
        updated_fields = []
        
        if 'system_prompt' in data:
            app_settings['system_prompt'] = data['system_prompt']
            updated_fields.append('system_prompt')
        
        if 'admin_system_prompt' in data:
            app_settings['admin_system_prompt'] = data['admin_system_prompt']
            updated_fields.append('admin_system_prompt')
        
        if not updated_fields:
            return jsonify({
                "success": False,
                "error": "No valid fields provided"
            }), 400
        
        if save_app_settings(app_settings):
            return jsonify({
                "success": True,
                "message": "System prompt updated successfully",
                "updated_fields": updated_fields
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to save settings"
            }), 500
        
    except Exception as e:
        logger.error(f"Error in update_system_prompt: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@settings_bp.route('/line-token', methods=['POST'])
def update_line_token():
    """Update Line Notify token"""
    try:
        data = request.get_json()
        
        if not data or 'token' not in data:
            return jsonify({
                "success": False,
                "error": "Token is required"
            }), 400
        
        token = data['token'].strip()
        
        # Basic token validation
        if token and not token.startswith('Bearer '):
            token = f"Bearer {token}"
        
        app_settings = load_app_settings()
        app_settings['line_notify_token'] = token
        
        if save_app_settings(app_settings):
            # Test token if provided
            if token:
                test_result = test_line_token(token)
                return jsonify({
                    "success": True,
                    "message": "Line token updated successfully",
                    "token_valid": test_result["valid"],
                    "test_message": test_result.get("message", "")
                })
            else:
                return jsonify({
                    "success": True,
                    "message": "Line token cleared successfully"
                })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to save token"
            }), 500
        
    except Exception as e:
        logger.error(f"Error in update_line_token: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@settings_bp.route('/line-token/test', methods=['POST'])
def test_line_notification():
    """Test Line notification"""
    try:
        app_settings = load_app_settings()
        token = app_settings.get('line_notify_token', '')
        
        if not token:
            return jsonify({
                "success": False,
                "error": "No Line token configured"
            }), 400
        
        test_result = test_line_token(token)
        return jsonify(test_result)
        
    except Exception as e:
        logger.error(f"Error in test_line_notification: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@settings_bp.route('/reset', methods=['POST'])
def reset_settings():
    """Reset settings to default"""
    try:
        default_settings = {
            "system_prompt": settings.default_system_prompt,
            "admin_system_prompt": settings.admin_system_prompt,
            "line_notify_token": "",
            "similarity_threshold": settings.similarity_threshold,
            "top_k_results": settings.top_k_results,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        if save_app_settings(default_settings):
            return jsonify({
                "success": True,
                "message": "Settings reset to default successfully",
                "settings": default_settings
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to reset settings"
            }), 500
        
    except Exception as e:
        logger.error(f"Error in reset_settings: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@settings_bp.route('/export', methods=['GET'])
def export_settings():
    """Export settings as JSON"""
    try:
        app_settings = load_app_settings()
        
        # Remove sensitive data for export
        export_data = app_settings.copy()
        if 'line_notify_token' in export_data:
            export_data['line_notify_token'] = "***HIDDEN***" if export_data['line_notify_token'] else ""
        
        return jsonify({
            "success": True,
            "settings": export_data,
            "exported_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in export_settings: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

def validate_settings(settings_data):
    """Validate settings data"""
    errors = []
    
    # Validate similarity threshold
    threshold = settings_data.get('similarity_threshold', 0.7)
    if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
        errors.append("similarity_threshold must be between 0 and 1")
    
    # Validate top_k_results
    top_k = settings_data.get('top_k_results', 5)
    if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
        errors.append("top_k_results must be between 1 and 20")
    
    # Validate chunk_size
    chunk_size = settings_data.get('chunk_size', 1000)
    if not isinstance(chunk_size, int) or chunk_size < 100 or chunk_size > 5000:
        errors.append("chunk_size must be between 100 and 5000")
    
    # Validate chunk_overlap
    chunk_overlap = settings_data.get('chunk_overlap', 200)
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0 or chunk_overlap > chunk_size // 2:
        errors.append("chunk_overlap must be between 0 and half of chunk_size")
    
    return errors

def test_line_token(token):
    """Test Line Notify token"""
    try:
        import requests
        
        url = "https://notify-api.line.me/api/notify"
        headers = {
            "Authorization": token,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "message": "🤖 OAI_BOT_V.1 Test Notification\nการทดสอบการแจ้งเตือนจากระบบ AI"
        }
        
        response = requests.post(url, headers=headers, data=data, timeout=10)
        
        if response.status_code == 200:
            return {
                "valid": True,
                "
