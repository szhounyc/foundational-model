#!/usr/bin/env python3
"""
Template Seeding Script for Contract Analysis System

This script seeds the application with predefined templates and provides
batch upload functionality for law firm templates.

Usage:
    python scripts/seed_templates.py --seed-default    # Seed with MNTN templates
    python scripts/seed_templates.py --batch-upload <law_firm_name> <source_dir>
    python scripts/seed_templates.py --list-templates  # List existing templates
"""

import os
import sys
import shutil
import sqlite3
import uuid
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add the parent directory to the path to import backend modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from backend.main import extract_text_from_file, is_supported_file
except ImportError:
    print("Warning: Could not import backend modules. Text extraction will be skipped.")
    def extract_text_from_file(file_path: str) -> str:
        return ""
    def is_supported_file(filename: str) -> bool:
        return filename.lower().endswith(('.pdf', '.docx'))

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
DATABASE_PATH = PROJECT_ROOT / "app.db"

# Template type mappings based on filename patterns
TEMPLATE_TYPE_PATTERNS = {
    'purchase_agreement': ['purchase agreement', 'purchase_agreement', 'agreement'],
    'rider': ['rider'],
    'legal_comments': ['comments', 'comment', 'legal comments', 'legal_comments']
}

# Default seed templates configuration
DEFAULT_SEED_TEMPLATES = {
    'law_firm_1': {
        'name': 'Marans Newman Tsolis & Nazinitsky LLC',
        'keywords': ['Marans Newman Tsolis', 'MNTN', 'Marans Newman', 'Tsolis', 'Nazinitsky'],
        'templates_source': 'dataset/zlg-re/MNTN/Radiant',
        'templates': [
            {
                'source_file': 'Unit 802 - 24-01 Queens Plaza North - Purchase Agreement (Yu and Wang).pdf',
                'template_type': 'purchase_agreement',
                'description': 'Standard purchase agreement template for residential properties'
            },
            {
                'source_file': '24-01 Queens Plaza North - Unit 802 (Yu and Wang) - Rider.pdf',
                'template_type': 'rider',
                'description': 'Standard rider template with additional terms and conditions'
            },
            {
                'source_file': 'Contract Comments MNTN Radiant.docx',
                'template_type': 'legal_comments',
                'description': 'Legal comments template for contract review guidance'
            }
        ]
    }
}

class TemplateSeeder:
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = DATA_DIR
        self.templates_dir = TEMPLATES_DIR
        self.db_path = DATABASE_PATH
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            self.data_dir,
            self.templates_dir,
            self.templates_dir / "purchase_agreement",
            self.templates_dir / "rider",
            self.templates_dir / "legal_comments",
            self.templates_dir / "main_contract"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Ensured directory exists: {directory}")
    
    def _get_db_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        if not self.db_path.exists():
            print(f"❌ Database not found at {self.db_path}")
            print("Please run the backend application first to initialize the database.")
            sys.exit(1)
        
        return sqlite3.connect(str(self.db_path))
    
    def _detect_template_type(self, filename: str) -> str:
        """Detect template type from filename"""
        filename_lower = filename.lower()
        
        for template_type, patterns in TEMPLATE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return template_type
        
        # Default fallback
        return 'main_contract'
    
    def _copy_template_file(self, source_path: Path, template_id: str, template_type: str, original_filename: str) -> Path:
        """Copy template file to the appropriate directory with proper naming"""
        # Get file extension
        file_extension = source_path.suffix
        
        # Create destination filename
        dest_filename = f"{template_id}_{original_filename}"
        dest_path = self.templates_dir / template_type / dest_filename
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        print(f"📄 Copied template: {source_path.name} -> {dest_path}")
        
        return dest_path
    
    def _extract_template_content(self, file_path: Path) -> str:
        """Extract text content from template file"""
        try:
            content = extract_text_from_file(str(file_path))
            print(f"📝 Extracted {len(content)} characters from {file_path.name}")
            return content
        except Exception as e:
            print(f"⚠️ Could not extract text from {file_path.name}: {e}")
            return ""
    
    def _insert_template_to_db(self, template_data: Dict) -> bool:
        """Insert template record into database"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO contract_templates (id, law_firm_id, law_firm_name, template_type, file_name, file_path, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                template_data['id'],
                template_data['law_firm_id'],
                template_data['law_firm_name'],
                template_data['template_type'],
                template_data['file_name'],
                template_data['file_path'],
                template_data['content'],
                template_data['created_at']
            ))
            
            conn.commit()
            conn.close()
            
            print(f"💾 Saved template to database: {template_data['file_name']}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save template to database: {e}")
            return False
    
    def _ensure_law_firm_exists(self, law_firm_id: str, law_firm_name: str, keywords: List[str]) -> bool:
        """Ensure law firm exists in database"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Check if law firm exists
            cursor.execute('SELECT id FROM law_firms WHERE id = ?', (law_firm_id,))
            if cursor.fetchone():
                print(f"✅ Law firm already exists: {law_firm_name}")
                conn.close()
                return True
            
            # Insert law firm
            cursor.execute('''
                INSERT INTO law_firms (id, name, keywords, created_at)
                VALUES (?, ?, ?, ?)
            ''', (law_firm_id, law_firm_name, json.dumps(keywords), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            print(f"🏢 Created law firm: {law_firm_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create law firm: {e}")
            return False
    
    def seed_default_templates(self) -> bool:
        """Seed the application with default MNTN templates"""
        print("🌱 Starting default template seeding...")
        
        success_count = 0
        total_count = 0
        
        for law_firm_id, firm_config in DEFAULT_SEED_TEMPLATES.items():
            print(f"\n🏢 Processing law firm: {firm_config['name']}")
            
            # Ensure law firm exists
            if not self._ensure_law_firm_exists(law_firm_id, firm_config['name'], firm_config['keywords']):
                continue
            
            # Process templates
            source_dir = self.project_root / firm_config['templates_source']
            if not source_dir.exists():
                print(f"❌ Source directory not found: {source_dir}")
                continue
            
            for template_config in firm_config['templates']:
                total_count += 1
                source_file = source_dir / template_config['source_file']
                
                if not source_file.exists():
                    print(f"❌ Template file not found: {source_file}")
                    continue
                
                if not is_supported_file(template_config['source_file']):
                    print(f"❌ Unsupported file type: {template_config['source_file']}")
                    continue
                
                # Generate template ID
                template_id = str(uuid.uuid4())
                
                # Copy file to templates directory
                try:
                    dest_path = self._copy_template_file(
                        source_file,
                        template_id,
                        template_config['template_type'],
                        template_config['source_file']
                    )
                    
                    # Extract content
                    content = self._extract_template_content(dest_path)
                    
                    # Prepare database record
                    template_data = {
                        'id': template_id,
                        'law_firm_id': law_firm_id,
                        'law_firm_name': firm_config['name'],
                        'template_type': template_config['template_type'],
                        'file_name': template_config['source_file'],
                        'file_path': str(dest_path.relative_to(self.project_root)),
                        'content': content,
                        'created_at': datetime.now().isoformat()
                    }
                    
                    # Save to database
                    if self._insert_template_to_db(template_data):
                        success_count += 1
                        print(f"✅ Successfully seeded template: {template_config['source_file']}")
                    
                except Exception as e:
                    print(f"❌ Failed to process template {template_config['source_file']}: {e}")
        
        print(f"\n🎉 Seeding completed: {success_count}/{total_count} templates successfully seeded")
        return success_count > 0
    
    def batch_upload_templates(self, law_firm_name: str, source_dir: str, law_firm_keywords: Optional[List[str]] = None) -> bool:
        """Batch upload templates from a directory for a specific law firm"""
        print(f"📦 Starting batch upload for law firm: {law_firm_name}")
        print(f"📁 Source directory: {source_dir}")
        
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"❌ Source directory not found: {source_dir}")
            return False
        
        # Generate law firm ID
        law_firm_id = f"law_firm_{law_firm_name.lower().replace(' ', '_').replace('&', 'and')}"
        
        # Use provided keywords or generate from name
        if not law_firm_keywords:
            law_firm_keywords = [law_firm_name, law_firm_name.split()[0]]
        
        # Ensure law firm exists
        if not self._ensure_law_firm_exists(law_firm_id, law_firm_name, law_firm_keywords):
            return False
        
        # Find all supported files in source directory
        supported_files = []
        for file_path in source_path.rglob('*'):
            if file_path.is_file() and is_supported_file(file_path.name):
                supported_files.append(file_path)
        
        if not supported_files:
            print(f"❌ No supported template files found in {source_dir}")
            return False
        
        print(f"📄 Found {len(supported_files)} template files")
        
        success_count = 0
        for file_path in supported_files:
            try:
                # Detect template type
                template_type = self._detect_template_type(file_path.name)
                
                # Generate template ID
                template_id = str(uuid.uuid4())
                
                # Copy file
                dest_path = self._copy_template_file(
                    file_path,
                    template_id,
                    template_type,
                    file_path.name
                )
                
                # Extract content
                content = self._extract_template_content(dest_path)
                
                # Prepare database record
                template_data = {
                    'id': template_id,
                    'law_firm_id': law_firm_id,
                    'law_firm_name': law_firm_name,
                    'template_type': template_type,
                    'file_name': file_path.name,
                    'file_path': str(dest_path.relative_to(self.project_root)),
                    'content': content,
                    'created_at': datetime.now().isoformat()
                }
                
                # Save to database
                if self._insert_template_to_db(template_data):
                    success_count += 1
                    print(f"✅ Uploaded: {file_path.name} ({template_type})")
                
            except Exception as e:
                print(f"❌ Failed to upload {file_path.name}: {e}")
        
        print(f"\n🎉 Batch upload completed: {success_count}/{len(supported_files)} templates uploaded")
        return success_count > 0
    
    def list_templates(self) -> bool:
        """List all existing templates in the database"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT ct.law_firm_name, ct.template_type, ct.file_name, ct.created_at, ct.id
                FROM contract_templates ct
                ORDER BY ct.law_firm_name, ct.template_type, ct.created_at
            ''')
            
            templates = cursor.fetchall()
            conn.close()
            
            if not templates:
                print("📭 No templates found in database")
                return True
            
            print(f"📋 Found {len(templates)} templates:")
            print("-" * 80)
            
            current_firm = None
            for law_firm_name, template_type, file_name, created_at, template_id in templates:
                if current_firm != law_firm_name:
                    current_firm = law_firm_name
                    print(f"\n🏢 {law_firm_name}")
                
                created_date = datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M")
                print(f"  📄 {template_type:20} | {file_name:40} | {created_date} | {template_id[:8]}...")
            
            print("-" * 80)
            return True
            
        except Exception as e:
            print(f"❌ Failed to list templates: {e}")
            return False
    
    def clean_templates(self, law_firm_id: Optional[str] = None) -> bool:
        """Clean templates (for development/testing)"""
        print("🧹 Cleaning templates...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            if law_firm_id:
                cursor.execute('SELECT file_path FROM contract_templates WHERE law_firm_id = ?', (law_firm_id,))
                cursor.execute('DELETE FROM contract_templates WHERE law_firm_id = ?', (law_firm_id,))
                print(f"🗑️ Cleaned templates for law firm: {law_firm_id}")
            else:
                cursor.execute('SELECT file_path FROM contract_templates')
                file_paths = cursor.fetchall()
                
                # Remove files
                for (file_path,) in file_paths:
                    try:
                        full_path = self.project_root / file_path
                        if full_path.exists():
                            full_path.unlink()
                    except Exception as e:
                        print(f"⚠️ Could not remove file {file_path}: {e}")
                
                cursor.execute('DELETE FROM contract_templates')
                print("🗑️ Cleaned all templates")
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to clean templates: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Template Seeding Script for Contract Analysis System')
    parser.add_argument('--seed-default', action='store_true', help='Seed with default MNTN templates')
    parser.add_argument('--batch-upload', nargs=2, metavar=('LAW_FIRM_NAME', 'SOURCE_DIR'), 
                       help='Batch upload templates for a law firm')
    parser.add_argument('--list-templates', action='store_true', help='List existing templates')
    parser.add_argument('--clean', action='store_true', help='Clean all templates (development only)')
    parser.add_argument('--keywords', nargs='*', help='Keywords for law firm (used with --batch-upload)')
    
    args = parser.parse_args()
    
    if not any([args.seed_default, args.batch_upload, args.list_templates, args.clean]):
        parser.print_help()
        return
    
    seeder = TemplateSeeder()
    
    if args.seed_default:
        seeder.seed_default_templates()
    
    if args.batch_upload:
        law_firm_name, source_dir = args.batch_upload
        keywords = args.keywords if args.keywords else None
        seeder.batch_upload_templates(law_firm_name, source_dir, keywords)
    
    if args.list_templates:
        seeder.list_templates()
    
    if args.clean:
        confirm = input("⚠️ This will delete all templates. Are you sure? (yes/no): ")
        if confirm.lower() == 'yes':
            seeder.clean_templates()

if __name__ == '__main__':
    main() 