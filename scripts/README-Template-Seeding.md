# Template Seeding System

This system provides automated seeding of contract templates for the Contract Analysis Backend. It includes both automatic seeding on startup and manual batch upload capabilities.

## 🚀 Quick Start

### Automatic Seeding
The system automatically seeds default templates for "Marans Newman Tsolis & Nazinitsky LLC" when starting up:

```bash
# Run the startup seeding script
./scripts/startup_seed.sh
```

### Manual Operations
```bash
# Seed default templates
python seed_templates.py --seed-default

# Batch upload templates for a law firm
python seed_templates.py --batch-upload "Law Firm Name" /path/to/templates

# List existing templates
python seed_templates.py --list-templates

# Clean templates (all or specific law firm)
python seed_templates.py --clean
python seed_templates.py --clean "Law Firm Name"
```

## 📁 Template Organization

### Template Types
The system automatically detects template types based on filename patterns:

- **purchase_agreement**: Files containing "purchase", "agreement", "sale"
- **rider**: Files containing "rider", "addendum", "amendment"  
- **legal_comments**: Files containing "comment", "review", "note"
- **main_contract**: Default for files that don't match other patterns

### Directory Structure
Templates are organized in the following structure:
```
data/
└── templates/
    ├── purchase_agreement/
    ├── rider/
    ├── legal_comments/
    └── main_contract/
```

## 🗄️ Database Schema

Templates are stored in the `contract_templates` table:

```sql
CREATE TABLE contract_templates (
    id TEXT PRIMARY KEY,
    law_firm_id TEXT NOT NULL,
    law_firm_name TEXT NOT NULL,
    template_type TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 📋 Default Templates

The system comes pre-configured with templates for **Marans Newman Tsolis & Nazinitsky LLC**:

1. **Legal Comments Template** (`legal_comments`)
   - Source: `dataset/zlg-re/MNTN/Radiant/Contract Comments MNTN Radiant.docx`
   - Description: Legal comments template for Radiant project

2. **Purchase Agreement Template** (`purchase_agreement`)
   - Source: `dataset/zlg-re/MNTN/Radiant/Unit 802 - 24-01 Queens Plaza North - Purchase Agreement (Yu and Wang).pdf`
   - Description: Purchase agreement template for Queens Plaza North

3. **Rider Template** (`rider`)
   - Source: `dataset/zlg-re/MNTN/Radiant/24-01 Queens Plaza North - Unit 802 (Yu and Wang) - Rider.pdf`
   - Description: Rider template for Queens Plaza North

## 🔧 Adding New Default Templates

To add new default templates, edit the `DEFAULT_SEED_TEMPLATES` configuration in `seed_templates.py`:

```python
DEFAULT_SEED_TEMPLATES = {
    "law_firm_name": "Your Law Firm Name",
    "law_firm_id": "your_law_firm_id",
    "templates": [
        {
            "source_path": "path/to/your/template.pdf",
            "template_type": "purchase_agreement",
            "description": "Description of your template"
        }
    ]
}
```

## 💼 Example Usage Scenarios

### Setting Up a New Law Firm
```bash
# Create templates directory for new firm
python seed_templates.py --batch-upload "Smith & Associates" /path/to/smith/templates

# Verify templates were uploaded
python seed_templates.py --list-templates
```

### Adding Templates to Existing Firm
```bash
# Upload additional templates
python seed_templates.py --batch-upload "Existing Firm" /path/to/new/templates

# Check what was added
python seed_templates.py --list-templates
```

### Cleaning Up Templates
```bash
# Remove all templates for a specific firm
python seed_templates.py --clean "Law Firm Name"

# Remove all templates (use with caution!)
python seed_templates.py --clean
```

## 📄 Supported File Types

- **PDF**: `.pdf`
- **Word Documents**: `.docx`, `.doc`
- **Text Files**: `.txt`

## 🔍 Template Detection Logic

The system uses intelligent filename pattern matching:

```python
TEMPLATE_TYPE_MAPPING = {
    'purchase_agreement': ['purchase', 'agreement', 'sale'],
    'rider': ['rider', 'addendum', 'amendment'],
    'legal_comments': ['comment', 'review', 'note'],
    'main_contract': ['contract', 'main', 'base']
}
```

Files are categorized based on keywords found in their filenames (case-insensitive).

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Missing source files**: Warns and continues with other templates
- **Database errors**: Logs errors and provides helpful messages
- **File copy failures**: Reports specific file issues
- **Invalid paths**: Validates directories before processing

## 🔗 Integration with Backend

### Startup Integration
The backend automatically checks for templates on startup. If none exist, it seeds the default templates.

### API Integration
Seeded templates are immediately available through the backend API:

```bash
# List templates via API
curl "http://localhost:9100/api/templates"

# Get specific template content
curl "http://localhost:9100/api/templates/{template_id}/content"
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Source file not found" errors**
   - Check that the source paths in `DEFAULT_SEED_TEMPLATES` are correct
   - Ensure the dataset directory exists and contains the expected files

2. **Database connection errors**
   - Verify the database directory exists: `backend/database/`
   - Check file permissions

3. **Templates not appearing in API**
   - Restart the backend service: `docker compose restart backend`
   - Check database contents: `python seed_templates.py --list-templates`

### Debug Mode
For detailed logging, the scripts use Python's logging module. Check the console output for detailed information about the seeding process.

## 🔮 Future Enhancements

- **Template validation**: Verify template content and structure
- **Bulk operations**: Import/export template configurations
- **Template versioning**: Track template changes over time
- **Content extraction**: Extract and index template text content
- **Template comparison**: Compare templates across law firms
- **API endpoints**: RESTful endpoints for template management

## 📞 Support

For issues or questions about the template seeding system:

1. Check the logs for detailed error messages
2. Verify file paths and permissions
3. Ensure the backend service is running
4. Review this documentation for common solutions

---

*Last updated: June 2025* 