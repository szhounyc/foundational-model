# Fireworks.ai Dataset Preparation for Contract Review

This directory contains tools for preparing contract review datasets for fine-tuning with Fireworks.ai. The script processes legal contracts (PDFs) and their corresponding review documents (Word) to create training data for LLM fine-tuning.

## Overview

The dataset processor handles training data where each project contains:
- **2 PDF files**: Contract documents with numbered sections
- **1 Word document**: Professional review comments referencing specific sections

The script extracts text, maps section references, and formats everything into Fireworks.ai's required JSONL format.

## Directory Structure

Your input data should be organized like this:

```
input_data/
├── project_001/
│   ├── contract_part1.pdf
│   ├── contract_part2.pdf
│   └── review_comments.docx
├── project_002/
│   ├── main_agreement.pdf
│   ├── appendix.pdf
│   └── legal_review.docx
└── project_003/
    ├── contract_a.pdf
    ├── contract_b.pdf
    └── review_notes.docx
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (if not already done):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Usage

### Basic Usage

```bash
python dataset_processor.py --input-dir /path/to/input_data --output-dir /path/to/output
```

### Advanced Usage

```bash
python dataset_processor.py \
    --input-dir ./sample_contracts \
    --output-dir ./processed_dataset \
    --output-filename contract_review_training.jsonl
```

### Parameters

- `--input-dir`: Directory containing project folders with PDFs and Word docs
- `--output-dir`: Directory where processed dataset will be saved
- `--output-filename`: Name of the output JSONL file (default: `fireworks_training_dataset.jsonl`)

## How It Works

### 1. Project Discovery
The script scans the input directory for folders containing:
- At least 2 PDF files
- At least 1 Word document (.docx or .doc)

### 2. Text Extraction
- **PDFs**: Uses PyMuPDF (fitz) for accurate text extraction, with PyPDF2 as fallback
- **Word docs**: Uses python-docx to extract text from paragraphs

### 3. Section Mapping
The script identifies numbered sections in PDFs using regex patterns:
- `1.1`, `2.3.4`, etc.
- `Section 1.1`, `Section 2.3.4`, etc.

### 4. Comment Processing
Review comments are parsed to:
- Extract section references (e.g., "3.1", "5.2")
- Map comments to specific contract sections
- Handle general comments without section references

### 5. Training Data Generation
Each review comment becomes a training example with:
- **System prompt**: Legal expert role definition
- **User prompt**: Contract section(s) to review
- **Assistant response**: The actual review comment

## Output Format

The script generates data in Fireworks.ai's chat format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a legal contract review expert..."
    },
    {
      "role": "user", 
      "content": "Please review section(s) 3.1 of this contract:\n\nSection 3.1:\n[contract text]"
    },
    {
      "role": "assistant",
      "content": "[review comment from Word document]"
    }
  ]
}
```

## Output Files

The script creates several files in the output directory:

1. **`fireworks_training_dataset.jsonl`**: Main training file for Fireworks.ai
2. **`fireworks_training_dataset.json`**: Same data in readable JSON format
3. **`{project_id}_metadata.json`**: Metadata for each project (debugging)

## Example Workflow

1. **Prepare your data**:
   ```
   contracts/
   ├── deal_001/
   │   ├── purchase_agreement.pdf
   │   ├── terms_conditions.pdf
   │   └── legal_review.docx
   ```

2. **Run the processor**:
   ```bash
   python dataset_processor.py --input-dir contracts --output-dir processed
   ```

3. **Check the output**:
   ```
   processed/
   ├── fireworks_training_dataset.jsonl
   ├── fireworks_training_dataset.json
   └── deal_001_metadata.json
   ```

4. **Upload to Fireworks.ai**:
   - Use the `.jsonl` file for fine-tuning
   - Follow Fireworks.ai documentation for creating training jobs

## Section Reference Handling

The script intelligently handles section references:

### Exact Matches
If a comment mentions "3.1" and section "3.1" exists, it's directly mapped.

### Partial Matches
If "3.1" is mentioned but only "3.1.1" exists, the script finds the closest match.

### General Comments
Comments without section references use multiple sections for context.

## Customization

### Modifying Section Patterns
Edit the `section_patterns` in `ContractSectionExtractor`:

```python
self.section_patterns = [
    r'(\d+\.\d+(?:\.\d+)?)\s+([A-Z][^.]*?)(?=\n\d+\.\d+|\n[A-Z]{2,}|\Z)',
    # Add your custom patterns here
]
```

### Adjusting System Prompt
Modify the `system_prompt` in `FireworksDatasetFormatter`:

```python
self.system_prompt = """Your custom system prompt for the legal expert role..."""
```

### Section Length Limits
Adjust the maximum section length in `extract_sections`:

```python
if len(section_text) > 2000:  # Change this limit
    section_text = section_text[:2000] + "..."
```

## Troubleshooting

### Common Issues

1. **No projects found**:
   - Check directory structure
   - Ensure each folder has 2+ PDFs and 1+ Word doc

2. **Poor text extraction**:
   - PDFs might be scanned images (need OCR)
   - Try different PDF processing tools

3. **Section references not found**:
   - Check if your contracts use different numbering schemes
   - Modify regex patterns accordingly

4. **Empty training examples**:
   - Verify Word documents contain section references
   - Check if section numbers match between PDFs and Word docs

### Debugging

Use the metadata files to debug:
```bash
cat processed/project_001_metadata.json | jq .
```

This shows:
- Sections found in PDFs
- Review comments extracted
- Section references identified

## Performance Tips

1. **Large datasets**: Process in batches to manage memory
2. **Complex PDFs**: Consider pre-processing with OCR tools
3. **Multiple formats**: Extend the script to handle other document types

## Integration with Fireworks.ai

1. **Upload dataset**: Use the generated `.jsonl` file
2. **Create fine-tuning job**: Follow Fireworks.ai documentation
3. **Monitor training**: Use Fireworks.ai dashboard
4. **Test model**: Deploy and test with new contracts

## License

This script is part of the LLM Fine-tuning Platform and follows the same MIT license. 