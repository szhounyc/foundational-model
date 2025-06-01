#!/usr/bin/env python3
"""
Template Seeding System for Contract Analysis Backend

This script seeds the application with predefined templates and provides
utilities for batch uploading templates.

Usage:
    python seed_templates.py --seed-default
    python seed_templates.py --batch-upload "Law Firm Name" /path/to/templates
    python seed_templates.py --list-templates
"""

import os
import sys
import shutil
import sqlite3
import uuid
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
DATABASE_PATH = PROJECT_ROOT / "app.db"

# Template type mappings based on filename patterns
TEMPLATE_TYPE_MAPPING = {
    'purchase_agreement': ['purchase', 'agreement', 'sale'],
    'rider': ['rider', 'addendum', 'amendment'],
    'legal_comments': ['comment', 'review', 'note'],
    'main_contract': ['contract', 'main', 'base']
}

# Default seed templates for MNTN law firm
DEFAULT_SEED_TEMPLATES = {
    "law_firm_name": "Marans Newman Tsolis & Nazinitsky LLC",
    "law_firm_id": "law_firm_1",
    "templates": [
        {
            "source_path": "../dataset/zlg-re/MNTN/Radiant/Contract Comments MNTN Radiant.docx",
            "template_type": "legal_comments",
            "description": "Legal comments template for Radiant project"
        },
        {
            "source_path": "../dataset/zlg-re/MNTN/Radiant/Unit 802 - 24-01 Queens Plaza North - Purchase Agreement (Yu and Wang).pdf",
            "template_type": "purchase_agreement", 
            "description": "Purchase agreement template for Queens Plaza North"
        },
        {
            "source_path": "../dataset/zlg-re/MNTN/Radiant/24-01 Queens Plaza North - Unit 802 (Yu and Wang) - Rider.pdf",
            "template_type": "rider",
            "description": "Rider template for Queens Plaza North"
        }
    ]
}


class TemplateSeeder:
    def __init__(self):
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        for template_type in TEMPLATE_TYPE_MAPPING.keys():
            (TEMPLATES_DIR / template_type).mkdir(exist_ok=True)
        
        # Ensure database directory exists
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    def get_db_connection(self) -> sqlite3.Connection:
        """Get database connection and ensure tables exist"""
        conn = sqlite3.connect('app.db')
        conn.row_factory = sqlite3.Row
        
        # Create contract_templates table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contract_templates (
                id TEXT PRIMARY KEY,
                law_firm_id TEXT NOT NULL,
                law_firm_name TEXT NOT NULL,
                template_type TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                content TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        return conn
    
    def detect_template_type(self, filename: str) -> str:
        """Detect template type based on filename patterns"""
        filename_lower = filename.lower()
        
        for template_type, keywords in TEMPLATE_TYPE_MAPPING.items():
            if any(keyword in filename_lower for keyword in keywords):
                return template_type
        
        # Default to main_contract if no pattern matches
        return 'main_contract'
    
    def copy_template_file(self, source_path: str, template_type: str) -> Tuple[str, str]:
        """Copy template file to appropriate directory and return new path and filename"""
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source template file not found: {source_path}")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = source_path.suffix
        new_filename = f"{file_id}_{source_path.name}"
        
        # Destination path
        dest_dir = TEMPLATES_DIR / template_type
        dest_path = dest_dir / new_filename
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        
        return str(dest_path.relative_to(PROJECT_ROOT)), new_filename
    
    def extract_content(self, file_path: str) -> str:
        """Extract text content from template file"""
        try:
            # Import the text extraction functions from main.py
            sys.path.append(str(PROJECT_ROOT))
            from main import extract_text_from_file
            return extract_text_from_file(file_path)
        except Exception as e:
            logger.warning(f"Could not extract text from {file_path}: {e}")
            return f"Content from {Path(file_path).name}"
    
    def insert_template_to_db(self, template_id: str, law_firm_id: str, law_firm_name: str,
                            template_type: str, file_name: str, file_path: str):
        """Insert template record into database"""
        # Extract content from the file
        content = self.extract_content(file_path)
        
        conn = self.get_db_connection()
        try:
            conn.execute("""
                INSERT INTO contract_templates 
                (id, law_firm_id, law_firm_name, template_type, file_name, file_path, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (template_id, law_firm_id, law_firm_name, template_type, file_name, file_path, content, datetime.now().isoformat()))
            conn.commit()
            logger.info(f"✅ Inserted template {file_name} into database")
        finally:
            conn.close()
    
    def law_firm_exists(self, law_firm_name: str) -> bool:
        """Check if law firm already has templates"""
        conn = self.get_db_connection()
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM contract_templates WHERE law_firm_name = ?",
                (law_firm_name,)
            )
            count = cursor.fetchone()[0]
            return count > 0
        finally:
            conn.close()
    
    def seed_default_templates(self):
        """Seed the application with default MNTN templates"""
        law_firm_name = DEFAULT_SEED_TEMPLATES["law_firm_name"]
        law_firm_id = DEFAULT_SEED_TEMPLATES["law_firm_id"]
        
        logger.info(f"🌱 Seeding default templates for {law_firm_name}...")
        
        if self.law_firm_exists(law_firm_name):
            logger.info(f"⚠️  Templates for {law_firm_name} already exist. Skipping seeding.")
            return
        
        success_count = 0
        for template_config in DEFAULT_SEED_TEMPLATES["templates"]:
            try:
                source_path = PROJECT_ROOT / template_config["source_path"]
                template_type = template_config["template_type"]
                
                if not source_path.exists():
                    logger.warning(f"⚠️  Source file not found: {source_path}")
                    continue
                
                # Copy file and get new path
                file_path, file_name = self.copy_template_file(str(source_path), template_type)
                
                # Generate template ID
                template_id = str(uuid.uuid4())
                
                # Insert into database
                self.insert_template_to_db(
                    template_id, law_firm_id, law_firm_name,
                    template_type, file_name, file_path
                )
                
                success_count += 1
                logger.info(f"✅ Seeded template: {file_name} ({template_type})")
                
            except Exception as e:
                logger.error(f"❌ Failed to seed template {template_config['source_path']}: {e}")
        
        logger.info(f"🎉 Successfully seeded {success_count} templates for {law_firm_name}")
    
    def batch_upload_templates(self, law_firm_name: str, source_dir: str):
        """Batch upload templates from a directory"""
        source_path = Path(source_dir)
        
        if not source_path.exists():
            logger.error(f"❌ Source directory not found: {source_dir}")
            return
        
        logger.info(f"📁 Batch uploading templates from {source_dir} for {law_firm_name}...")
        
        # Generate law firm ID (simple slug)
        law_firm_id = law_firm_name.lower().replace(" ", "_").replace("&", "and")
        
        success_count = 0
        supported_extensions = {'.pdf', '.docx', '.doc', '.txt'}
        
        for file_path in source_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    # Detect template type
                    template_type = self.detect_template_type(file_path.name)
                    
                    # Copy file
                    new_file_path, new_filename = self.copy_template_file(str(file_path), template_type)
                    
                    # Generate template ID
                    template_id = str(uuid.uuid4())
                    
                    # Insert into database
                    self.insert_template_to_db(
                        template_id, law_firm_id, law_firm_name,
                        template_type, new_filename, new_file_path
                    )
                    
                    success_count += 1
                    logger.info(f"✅ Uploaded: {new_filename} ({template_type})")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to upload {file_path.name}: {e}")
        
        logger.info(f"🎉 Successfully uploaded {success_count} templates for {law_firm_name}")
    
    def list_templates(self):
        """List all templates in the database"""
        conn = self.get_db_connection()
        try:
            cursor = conn.execute("""
                SELECT law_firm_name, template_type, file_name, created_at
                FROM contract_templates
                ORDER BY law_firm_name, template_type, created_at
            """)
            
            templates = cursor.fetchall()
            
            if not templates:
                logger.info("📋 No templates found in database")
                return
            
            logger.info("📋 Current templates in database:")
            current_firm = None
            
            for template in templates:
                if template['law_firm_name'] != current_firm:
                    current_firm = template['law_firm_name']
                    logger.info(f"\n🏢 {current_firm}:")
                
                logger.info(f"  📄 {template['file_name']} ({template['template_type']}) - {template['created_at']}")
                
        finally:
            conn.close()
    
    def clean_templates(self, law_firm_name: Optional[str] = None):
        """Clean templates from database and filesystem"""
        conn = self.get_db_connection()
        try:
            if law_firm_name:
                logger.info(f"🧹 Cleaning templates for {law_firm_name}...")
                cursor = conn.execute(
                    "SELECT file_path FROM contract_templates WHERE law_firm_name = ?",
                    (law_firm_name,)
                )
                templates = cursor.fetchall()
                
                conn.execute("DELETE FROM contract_templates WHERE law_firm_name = ?", (law_firm_name,))
            else:
                logger.info("🧹 Cleaning all templates...")
                cursor = conn.execute("SELECT file_path FROM contract_templates")
                templates = cursor.fetchall()
                
                conn.execute("DELETE FROM contract_templates")
            
            conn.commit()
            
            # Remove files
            for template in templates:
                file_path = PROJECT_ROOT / template['file_path']
                if file_path.exists():
                    file_path.unlink()
            
            logger.info(f"✅ Cleaned {len(templates)} templates")
            
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Template Seeding System")
    parser.add_argument("--seed-default", action="store_true", 
                       help="Seed with default MNTN templates")
    parser.add_argument("--batch-upload", nargs=2, metavar=("LAW_FIRM", "SOURCE_DIR"),
                       help="Batch upload templates for a law firm")
    parser.add_argument("--list-templates", action="store_true",
                       help="List all templates in database")
    parser.add_argument("--clean", nargs="?", const="", metavar="LAW_FIRM",
                       help="Clean templates (optionally for specific law firm)")
    
    args = parser.parse_args()
    
    seeder = TemplateSeeder()
    
    if args.seed_default:
        seeder.seed_default_templates()
    elif args.batch_upload:
        law_firm_name, source_dir = args.batch_upload
        seeder.batch_upload_templates(law_firm_name, source_dir)
    elif args.list_templates:
        seeder.list_templates()
    elif args.clean is not None:
        law_firm_name = args.clean if args.clean else None
        seeder.clean_templates(law_firm_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 