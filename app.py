"""
OAI_BOT_V.1 - AI Document Analyzer
Power by NT AI ONE

Main Flask Application
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from datetime import datetime
import uuid

# Import services
from config.settings import Settings
from core.ai_service import AIService
from core.document_processor import DocumentProcessor
from core.rag_engine import RAGEngine
from api.documents import documents_bp
from api.chat import chat_bp
from api.settings import settings_bp
from api.admin import admin_bp

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Enable CORS
CORS(app)

# Initialize services
settings = Settings()
ai_service = AIService(settings)
doc_processor = DocumentProcessor(settings)
rag_engine = RAGEngine(settings)

# Register blueprints
app.register_blueprint(documents_bp, url_prefix='/api/documents')
app.register_blueprint(chat_bp, url_prefix='/api/chat')
app.register_blueprint(settings_bp, url_prefix='/api/settings')
app.register_blueprint(admin_bp, url_prefix='/api/admin')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """หน้าหลัก - คู่มือการใช้งาน"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """หน้า Dashboard หลัก"""
    return render_template('dashboard.html')

@app.route('/chat')
def chat():
    """หน้า Chat Interface"""
    return render_template('chat.html')

@app.route('/documents')
def documents():
    """หน้าจัดการเอกสาร"""
    return render_template('documents.html')

@app.route('/settings')
def settings_page():
    """หน้า Settings"""
    return render_template('settings.html')

@app.route('/admin')
def admin():
    """หน้า Admin Helper"""
    return render_template('admin.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {
            'ai_service': ai_service.health_check(),
            'qdrant': rag_engine.health_check(),
            'document_processor': doc_processor.health_check()
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not Found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting OAI_BOT_V.1 on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
