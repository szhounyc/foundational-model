# Template Seeding System

This directory contains scripts for seeding the contract analysis system with predefined templates and managing batch uploads of law firm templates.

## Overview

The template seeding system provides:

1. **Automatic seeding** of default templates on application startup
2. **Manual seeding** of predefined templates
3. **Batch upload** functionality for law firm templates
4. **Template management** utilities

## Files

- `seed_templates.py` - Main seeding script with comprehensive functionality
- `auto_seed_on_startup.py` - Automatic seeding on backend startup
- `README_TEMPLATE_SEEDING.md` - This documentation

## Quick Start

### Automatic Seeding (Default)

The system automatically seeds default templates when the backend starts if no templates exist in the database. This includes:

- **Marans Newman Tsolis & Nazinitsky LLC** templates:
  - Purchase Agreement
  - Rider
  - Legal Comments

### Manual Operations

#### 1. Seed Default Templates

```bash
python scripts/seed_templates.py --seed-default
```

This will:
- Copy templates from `dataset/zlg-re/MNTN/Radiant/` to `data/templates/`
- Extract text content from PDF/DOCX files
- Add templates to the database with proper metadata
- Organize files by template type (purchase_agreement, rider, legal_comments)

#### 2. Batch Upload Templates for a Law Firm

```bash
python scripts/seed_templates.py --batch-upload "Smith & Associates" "/path/to/templates"
```

Optional with custom keywords:
```bash
python scripts/seed_templates.py --batch-upload "Smith & Associates" "/path/to/templates" --keywords "Smith" "Associates" "S&A"
```

This will:
- Create the law firm in the database if it doesn't exist
- Auto-detect template types from filenames
- Copy and organize all PDF/DOCX files from the source directory
- Extract text content and save to database

#### 3. List Existing Templates

```bash
python scripts/seed_templates.py --list-templates
```

#### 4. Clean Templates (Development)

```bash
python scripts/seed_templates.py --clean
```

**⚠️ Warning**: This will delete all templates and their files!

## Template Type Detection

The system automatically detects template types based on filename patterns:

| Template Type | Filename Patterns |
|---------------|-------------------|
| `purchase_agreement` | "purchase agreement", "purchase_agreement", "agreement" |
| `rider` | "rider" |
| `legal_comments` | "comments", "comment", "legal comments", "legal_comments" |

If no pattern matches, files are categorized as `main_contract`.

## Directory Structure

Templates are organized in the following structure:

```
data/
└── templates/
    ├── purchase_agreement/
    │   └── {template_id}_{original_filename}
    ├── rider/
    │   └── {template_id}_{original_filename}
    ├── legal_comments/
    │   └── {template_id}_{original_filename}
    └── main_contract/
        └── {template_id}_{original_filename}
```

## Database Schema

Templates are stored in the `contract_templates` table:

```sql
CREATE TABLE contract_templates (
    id TEXT PRIMARY KEY,
    law_firm_id TEXT NOT NULL,
    law_firm_name TEXT NOT NULL,
    template_type TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    content TEXT,
    created_at TEXT NOT NULL
);
```

## Adding New Default Templates

To add new default templates, edit the `DEFAULT_SEED_TEMPLATES` configuration in `seed_templates.py`:

```python
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
            # Add more templates here...
        ]
    },
    # Add new law firms here...
}
```

## Example Usage Scenarios

### Scenario 1: Setting up a new law firm

```bash
# Create directory structure for new firm templates
mkdir -p /tmp/new_firm_templates

# Copy templates to the directory
cp /path/to/firm/templates/* /tmp/new_firm_templates/

# Batch upload
python scripts/seed_templates.py --batch-upload "Johnson & Partners" "/tmp/new_firm_templates" --keywords "Johnson" "Partners" "J&P"
```

### Scenario 2: Adding templates to existing firm

```bash
# Upload additional templates to existing firm
python scripts/seed_templates.py --batch-upload "Marans Newman Tsolis & Nazinitsky LLC" "/path/to/additional/templates"
```

### Scenario 3: Development workflow

```bash
# Clean all templates
python scripts/seed_templates.py --clean

# Re-seed with defaults
python scripts/seed_templates.py --seed-default

# Check what was seeded
python scripts/seed_templates.py --list-templates
```

## Supported File Types

- **PDF** (.pdf)
- **DOCX** (.docx)

## Error Handling

The seeding system includes comprehensive error handling:

- **Missing source files**: Skipped with warning
- **Unsupported file types**: Skipped with warning
- **Text extraction failures**: File is still saved but without content
- **Database errors**: Detailed error messages with rollback
- **Permission errors**: Clear error messages

## Integration with Backend

The seeding system is integrated with the backend through:

1. **Automatic startup seeding**: `auto_seed_on_startup.py` is called during backend startup
2. **Shared utilities**: Uses the same text extraction functions as the main application
3. **Database compatibility**: Uses the same database schema and connection methods

## Troubleshooting

### Common Issues

1. **"Database not found"**
   - Ensure the backend has been started at least once to initialize the database

2. **"Source directory not found"**
   - Check that the path to template files is correct
   - Ensure you're running from the project root directory

3. **"Text extraction failed"**
   - Verify that PDF/DOCX files are not corrupted
   - Check that required dependencies (PyPDF2, python-docx) are installed

4. **"Permission denied"**
   - Ensure write permissions to the `data/templates/` directory
   - Check file permissions on source template files

### Debug Mode

For detailed logging, you can modify the scripts to include debug output or run with Python's verbose mode:

```bash
python -v scripts/seed_templates.py --seed-default
```

## Future Enhancements

Potential improvements to the seeding system:

1. **Template validation**: Verify template content quality
2. **Incremental updates**: Update existing templates without full replacement
3. **Template versioning**: Track template versions and changes
4. **Bulk operations**: API endpoints for bulk template management
5. **Template metadata**: Additional fields like tags, categories, descriptions
6. **Import/export**: JSON-based template configuration files 