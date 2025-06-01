#!/usr/bin/env python3
"""
Auto-seed templates on startup

This script is called during backend startup to automatically seed
templates if the database is empty.
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.seed_templates import TemplateSeeder

def should_auto_seed() -> bool:
    """Check if we should auto-seed templates"""
    try:
        db_path = Path(__file__).parent.parent / "app.db"
        if not db_path.exists():
            return False
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM contract_templates')
        count = cursor.fetchone()[0]
        conn.close()
        
        return count == 0
    except Exception:
        return False

def auto_seed_templates():
    """Auto-seed templates if database is empty"""
    if should_auto_seed():
        print("🌱 Database is empty, auto-seeding default templates...")
        seeder = TemplateSeeder()
        success = seeder.seed_default_templates()
        if success:
            print("✅ Auto-seeding completed successfully")
        else:
            print("⚠️ Auto-seeding failed, but application will continue")
    else:
        print("📋 Templates already exist, skipping auto-seed")

if __name__ == '__main__':
    auto_seed_templates() 