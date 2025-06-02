#!/bin/bash

# Startup Template Seeding Script
# This script ensures that default templates are seeded when the system starts

echo "🌱 Checking if templates need to be seeded..."

# Check if we're in the correct directory
if [ ! -f "seed_templates.py" ]; then
    echo "❌ Error: seed_templates.py not found. Please run this script from the project root."
    exit 1
fi

# Check if templates already exist
TEMPLATE_COUNT=$(python -c "
import sqlite3
import sys
import os
from pathlib import Path

db_path = os.environ.get('DATABASE_PATH', 'app.db')
if not Path(db_path).exists():
    print('0')
    sys.exit()

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute('SELECT COUNT(*) FROM contract_templates WHERE law_firm_name = ?', ('Marans Newman Tsolis & Nazinitsky LLC',))
    count = cursor.fetchone()[0]
    conn.close()
    print(count)
except:
    print('0')
")

if [ "$TEMPLATE_COUNT" -gt 0 ]; then
    echo "✅ Templates already exist for Marans Newman Tsolis & Nazinitsky LLC ($TEMPLATE_COUNT templates found)"
    echo "📋 Listing existing templates:"
    python seed_templates.py --list-templates
else
    echo "🌱 No templates found, seeding default templates..."
    python seed_templates.py --seed-default
    
    if [ $? -eq 0 ]; then
        echo "✅ Template seeding completed successfully!"
        echo "📋 Listing seeded templates:"
        python seed_templates.py --list-templates
    else
        echo "❌ Template seeding failed!"
        exit 1
    fi
fi

echo "🎉 Template seeding check complete!" 