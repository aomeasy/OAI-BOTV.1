"""
Admin API endpoints for OAI_BOT_V.1
"""

import logging
from flask import Blueprint, request, jsonify
from datetime import datetime

from core.ai_service import AIService
from core.rag_engine import RAGEngine
from config.settings import Settings

logger = logging.getLogger(__name__)

# Create blueprint
admin_bp = Blueprint('admin', __name__)

# Initialize services
settings = Settings()
ai_service = AIService(settings)
rag_engine = RAGEngine(settings)

# Load admin settings
def load_admin_settings():
    try:
        from api.settings import load_app_settings
        return load_app_settings()
    except:
        return {"admin_system_prompt": settings.admin_system_prompt}

@admin_bp.route('/chat', methods=['POST'])
async def admin_chat():
    """Chat with admin assistant"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "Message is required"
            }), 400
        
        message = data['message']
        session_id = data.get('session_id', 'admin_default')
        
        # Get admin system prompt
        admin_settings = load_admin_settings()
        admin_prompt = admin_settings.get('admin_system_prompt', settings.admin_system_prompt)
        
        # Enhanced admin prompt with system info
        enhanced_admin_prompt = f"""{admin_prompt}

ข้อมูลระบบปัจจุบัน:
- ชื่อระบบ: OAI_BOT_V.1
- องค์กร: บลนป. (บริษัทลิสซิ่งนครหลวง จำกัด)
- ฟีเจอร์หลัก: RAG Document Analysis, AI Chatbot, OCR ภาษาไทย
- รองรับไฟล์: PDF, DOC, DOCX, TXT
- AI Models: Qwen3:14b, nomic-embed-text, Typhoon models
- Vector Database: Qdrant Cloud

คำแนะนำการใช้งาน:
1. อัพโหลดเอกสารที่หน้า "เอกสาร"
2. รอระบบประมวลผล (1-3 นาที)
3. ใช้ Chat เพื่อถามคำถามเกี่ยวกับเอกสาร
4. ตั้งค่า System Prompt ที่หน้า "ตั้งค่า"
5. กำหนด Line Token สำหรับการแจ้งเตือน

ตอบคำถามในรูปแบบที่เป็นมิตรและให้ขั้นตอนที่ชัดเจน
"""
        
        # Prepare messages
        messages = [{"role": "user", "content": message}]
        
        # Generate response
        response_result = await ai_service.generate_chat_response(
            messages=messages,
            system_prompt=enhanced_admin_prompt
        )
        
        if not response_result["success"]:
            return jsonify(response_result), 500
        
        return jsonify({
            "success": True,
            "response": response_result["response"],
            "session_id": session_id,
            "model_used": response_result.get("model_used"),
            "timestamp": datetime.now().isoformat(),
            "type": "admin_assistance"
        })
        
    except Exception as e:
        logger.error(f"Error in admin_chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@admin_bp.route('/help/getting-started', methods=['GET'])
def get_getting_started():
    """Get getting started guide"""
    try:
        guide = {
            "title": "คู่มือเริ่มต้นใช้งาน OAI_BOT_V.1",
            "steps": [
                {
                    "step": 1,
                    "title": "อัพโหลดเอกสาร",
                    "description": "ไปที่หน้า 'เอกสาร' และอัพโหลดไฟล์ที่ต้องการวิเคราะห์",
                    "details": [
                        "รองรับไฟล์: PDF, DOC, DOCX, TXT",
                        "ขนาดไฟล์สูงสุด: 50 MB",
                        "ระบบจะทำ OCR อัตโนมัติสำหรับภาษาไทย"
                    ]
                },
                {
                    "step": 2,
                    "title": "รอการประมวลผล",
                    "description": "ระบบจะประมวลผลเอกสารโดยอัตโนมัติ",
                    "details": [
                        "ใช้เวลาประมาณ 1-3 นาที",
                        "ระบบจะแยกข้อความและสร้าง Embedding",
                        "เก็บข้อมูลใน Vector Database"
                    ]
                },
                {
                    "step": 3,
                    "title": "เริ่มใช้งาน Chat",
                    "description": "ไปที่หน้า 'แชท' เพื่อถามคำถามเกี่ยวกับเอกสาร",
                    "details": [
                        "ระบบจะค้นหาข้อมูลที่เกี่ยวข้อง",
                        "ตอบคำถามโดยอ้างอิงจากเอกสาร",
                        "รองรับการสนทนาต่อเนื่อง"
                    ]
                },
                {
                    "step": 4,
                    "title": "ปรับแต่งการตั้งค่า",
                    "description": "ตั้งค่าระบบให้เหมาะกับการใช้งาน",
                    "details": [
                        "กำหนด System Prompt",
                        "ตั้งค่า Line Token สำหรับแจ้งเตือน",
                        "ปรับค่าความแม่นยำการค้นหา"
                    ]
                }
            ],
            "tips": [
                "ถามคำถามที่เฉพาะเจาะจงจะได้คำตอบที่แม่นยำ",
                "ใช้คำศัพท์ที่มีในเอกสารเพื่อผลลัพธ์ที่ดีกว่า",
                "ตรวจสอบ Sources ที่แสดงเพื่อยืนยันข้อมูล",
                "ลองใช้ Admin Helper หากต้องการความช่วยเหลือ"
            ]
        }
        
        return jsonify({
            "success": True,
            "guide": guide
        })
        
    except Exception as e:
        logger.error(f"Error in get_getting_started: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@admin_bp.route('/help/troubleshooting', methods=['GET'])
def get_troubleshooting():
    """Get troubleshooting guide"""
    try:
        troubleshooting = {
            "title": "การแก้ไขปัญหาทั่วไป",
            "issues": [
                {
                    "problem": "ไม่สามารถอัพโหลดไฟล์ได้",
                    "solutions": [
                        "ตรวจสอบขนาดไฟล์ (ต้องไม่เกิน 50 MB)",
                        "ตรวจสอบประเภทไฟล์ (PDF, DOC, DOCX, TXT)",
                        "ลองเปลี่ยนชื่อไฟล์เป็นภาษาอังกฤษ",
                        "ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต"
                    ]
                },
                {
                    "problem": "AI ตอบไม่ตรงคำถาม",
                    "solutions": [
                        "ตรวจสอบว่าเอกสารมีข้อมูลที่เกี่ยวข้อง",
                        "ลองเปลี่ยนวิธีการถามคำถาม",
                        "ปรับค่า Similarity Threshold ในหน้าตั้งค่า",
                        "ตรวจสอบ System Prompt ว่าเหมาะสม"
                    ]
                },
                {
                    "problem": "ระบบช้าหรือไม่ตอบสนอง",
                    "solutions": [
                        "รอสักครู่ ระบบอาจกำลังประมวลผล",
                        "ลองรีเฟรชหน้าเว็บ",
                        "ตรวจสอบสถานะระบบที่แถบด้านบน",
                        "ลองลดขนาดไฟล์หรือแบ่งเอกสาร"
                    ]
                },
                {
                    "problem": "การแจ้งเตือน Line ไม่ทำงาน",
                    "solutions": [
                        "ตรวจสอบ Line Token ที่หน้าตั้งค่า",
                        "ทดสอบการส่งข้อความด้วยปุ่ม Test",
                        "ตรวจสอบว่า Token ยังไม่หมดอายุ",
                        "ลองสร้าง Token ใหม่จาก LINE Notify"
                    ]
                }
            ],
            "contact": {
                "admin_chat": "ใช้ Admin Chatbot ในหน้านี้",
                "system_status": "ตรวจสอบที่ /health",
                "documentation": "ดูคู่มือที่หน้าหลัก"
            }
        }
        
        return jsonify({
            "success": True,
            "troubleshooting": troubleshooting
        })
        
    except Exception as e:
        logger.error(f"Error in get_troubleshooting: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@admin_bp.route('/help/questions', methods=['GET'])
def get_question_templates():
    """Get question templates and suggestions"""
    try:
        templates = {
            "title": "ตัวอย่างคำถามที่ดี",
            "categories": [
                {
                    "category": "การสรุปเอกสาร",
                    "examples": [
                        "สรุปเนื้อหาสำคัญของเอกสารนี้",
                        "จุดประสงค์หลักของเอกสารคืออะไร",
                        "มีข้อมูลสำคัญอะไรบ้างในเอกสาร"
                    ]
                },
                {
                    "category": "การค้นหาข้อมูลเฉพาะ",
                    "examples": [
                        "ในเอกสารมีการกล่าวถึง [หัวข้อ] อย่างไร",
                        "ขั้นตอนการ [กระบวนการ] คืออะไร",
                        "ข้อกำหนดสำหรับ [เรื่อง] มีอะไรบ้าง"
                    ]
                },
                {
                    "category": "การเปรียบเทียบและวิเคราะห์",
                    "examples": [
                        "ความแตกต่างระหว่าง A กับ B คืออะไร",
                        "ข้อดีข้อเสียของวิธีการนี้",
                        "แนวทางไหนเหมาะสมกับสถานการณ์นี้"
                    ]
                },
                {
                    "category": "การประยุกต์ใช้งาน",
                    "examples": [
                        "จะนำข้อมูลนี้ไปใช้ในงานได้อย่างไร",
                        "มีตัวอย่างการใช้งานจริงหรือไม่",
                        "ควรปฏิบัติตามขั้นตอนไหนก่อน"
                    ]
                }
            ],
            "tips": [
                "ใช้คำศัพท์ที่เฉพาะเจาะจง",
                "ระบุบริบทของคำถาม",
                "ถามทีละประเด็น",
                "ใช้ประโยคคำถามที่ชัดเจน"
            ]
        }
        
        return jsonify({
            "success": True,
            "templates": templates
        })
        
    except Exception as e:
        logger.error(f"Error in get_question_templates: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@admin_bp.route('/system/status', methods=['GET'])
async def get_system_status():
    """Get detailed system status"""
    try:
        # Get system stats from RAG engine
        stats_result = await rag_engine.get_system_stats()
        
        system_status = {
            "overall_health": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "ai_service": ai_service.health_check(),
                "rag_engine": rag_engine.health_check(),
                "vector_database": rag_engine.qdrant_service.health_check(),
                "document_processor": True  # Simplified check
            },
            "metrics": {},
            "version": "1.0.0"
        }
        
        if stats_result["success"]:
            system_status["metrics"] = stats_result["stats"]
        
        # Determine overall health
        if not all(system_status["components"].values()):
            system_status["overall_health"] = "degraded"
        
        return jsonify({
            "success": True,
            "status": system_status
        })
        
    except Exception as e:
        logger.error(f"Error in get_system_status: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@admin_bp.route('/system/info', methods=['GET'])
def get_system_info():
    """Get system information"""
    try:
        system_info = {
            "application": {
                "name": "OAI_BOT_V.1",
                "version": "1.0.0",
                "description": "AI Document Analyzer for บลนป.",
                "powered_by": "NT AI ONE"
            },
            "ai_models": {
                "chat_model": settings.chat_model,
                "embedding_model": settings.embedding_model,
                "thai_llm": settings.thai_llm_model,
                "thai_ocr": settings.thai_ocr_model
            },
            "infrastructure": {
                "vector_database": "Qdrant Cloud",
                "backend": "Python Flask",
                "deployment": "Render.com"
            },
            "features": [
                "Document Upload & Processing",
                "RAG-based Q&A",
                "Thai OCR Support",
                "Admin Assistant",
                "Line Notifications",
                "Custom System Prompts"
            ],
            "supported_formats": settings.allowed_extensions,
            "max_file_size_mb": settings.max_file_size / (1024 * 1024)
        }
        
        return jsonify({
            "success": True,
            "info": system_info
        })
        
    except Exception as e:
        logger.error(f"Error in get_system_info: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500
