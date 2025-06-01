#!/usr/bin/env python3
"""
Example usage of the Fireworks.ai dataset processor

This script demonstrates how to use the dataset processor with sample data.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from dataset_processor import DatasetProcessor
import config

def main():
    """Example usage of the dataset processor"""
    
    print("Fireworks.ai Dataset Processor - Example Usage")
    print("=" * 50)
    
    # Example 1: Basic usage
    print("\n1. Basic Usage:")
    print("python dataset_processor.py --input-dir ./contracts --output-dir ./processed")
    
    # Example 2: Custom filename
    print("\n2. Custom Output Filename:")
    print("python dataset_processor.py \\")
    print("    --input-dir ./sample_contracts \\")
    print("    --output-dir ./training_data \\")
    print("    --output-filename my_contract_dataset.jsonl")
    
    # Example 3: Directory structure
    print("\n3. Expected Directory Structure:")
    print("""
    input_contracts/
    ├── project_001/
    │   ├── main_contract.pdf
    │   ├── appendix.pdf
    │   └── review_comments.docx
    ├── project_002/
    │   ├── service_agreement.pdf
    │   ├── terms_conditions.pdf
    │   └── legal_review.docx
    └── project_003/
        ├── purchase_order.pdf
        ├── specifications.pdf
        └── review_notes.docx
    """)
    
    # Example 4: Configuration options
    print("\n4. Configuration Options:")
    print("You can customize the processor by editing config.py:")
    print(f"- Section patterns: {len(config.SECTION_PATTERNS)} patterns defined")
    print(f"- System prompts: {len(config.SYSTEM_PROMPTS)} contract types supported")
    print(f"- Max section length: {config.TEXT_PROCESSING['max_section_length']} characters")
    
    # Example 5: Output format
    print("\n5. Output Format (Fireworks.ai JSONL):")
    print("""
    {
      "messages": [
        {
          "role": "system",
          "content": "You are a legal contract review expert..."
        },
        {
          "role": "user",
          "content": "Please review section(s) 3.1 of this contract:\\n\\nSection 3.1:\\n[contract text]"
        },
        {
          "role": "assistant",
          "content": "[review comment from Word document]"
        }
      ]
    }
    """)
    
    # Example 6: Integration with Fireworks.ai
    print("\n6. Integration with Fireworks.ai:")
    print("1. Upload the generated .jsonl file to Fireworks.ai")
    print("2. Create a fine-tuning job using the uploaded dataset")
    print("3. Monitor training progress in the Fireworks.ai dashboard")
    print("4. Deploy the fine-tuned model for contract review")
    
    # Example 7: Quality tips
    print("\n7. Tips for Better Results:")
    print("- Ensure PDFs have clear section numbering (1.1, 2.3, etc.)")
    print("- Word documents should reference specific sections")
    print("- Use consistent formatting across all documents")
    print("- Review generated metadata files for debugging")
    print("- Start with a small dataset to test the pipeline")
    
    # Example 8: Troubleshooting
    print("\n8. Common Issues and Solutions:")
    print("- 'No projects found': Check directory structure and file extensions")
    print("- 'Poor section extraction': Verify PDF text quality and numbering")
    print("- 'Missing section references': Ensure Word docs mention section numbers")
    print("- 'Empty training examples': Check metadata files for debugging info")
    
    print("\n" + "=" * 50)
    print("Ready to process your contract datasets!")
    print("Run the test script first: python test_processor.py")

if __name__ == "__main__":
    main() 