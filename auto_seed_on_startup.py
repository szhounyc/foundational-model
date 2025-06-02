#!/usr/bin/env python3
"""
Auto-seeding functionality for Contract Analysis Backend

This module automatically seeds templates when the backend starts
if the database is empty.
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def should_auto_seed() -> bool:
    """Check if we should auto-seed templates"""
    # Use the same DATABASE_PATH logic as the backend
    database_path = os.environ.get('DATABASE_PATH', 'app.db')
    db_path = Path(database_path)
    
    if not db_path.exists():
        return True
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM contract_templates")
        count = cursor.fetchone()[0]
        conn.close()
        return count == 0
    except sqlite3.OperationalError:
        # Table doesn't exist, so we should seed
        return True
    except Exception:
        # Any other error, don't seed to be safe
        return False

def auto_seed_templates():
    """Auto-seed templates if database is empty"""
    if should_auto_seed():
        print("🌱 Database is empty, auto-seeding default templates...")
        try:
            from seed_templates import TemplateSeeder
            seeder = TemplateSeeder()
            seeder.seed_default_templates()
            print("✅ Auto-seeding completed successfully")
        except Exception as e:
            print(f"❌ Auto-seeding failed: {e}")
    else:
        print("📋 Templates already exist, skipping auto-seeding")

if __name__ == "__main__":
    auto_seed_templates() 