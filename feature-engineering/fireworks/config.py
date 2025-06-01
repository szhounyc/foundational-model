"""
Configuration file for Fireworks.ai dataset processor
"""

# Section extraction patterns
SECTION_PATTERNS = [
    # Standard numbered sections: 1.1, 2.3.4, etc.
    r'(\d+\.\d+(?:\.\d+)?)\s+([A-Z][^.]*?)(?=\n\d+\.\d+|\n[A-Z]{2,}|\Z)',
    
    # Sections with "Section" prefix: Section 1.1, Section 2.3.4
    r'(Section\s+\d+\.\d+(?:\.\d+)?)[:\s]+([^.]*?)(?=\nSection|\n[A-Z]{2,}|\Z)',
    
    # Sections with colon: 1.1: Title, 2.3.4: Description
    r'(\d+\.\d+(?:\.\d+)?)[:\s]*([A-Z][^.]*?)(?=\n\d+\.\d+|\n[A-Z]{2,}|\Z)',
    
    # Article sections: Article 1, Article II
    r'(Article\s+[IVX]+|\d+)[:\s]*([A-Z][^.]*?)(?=\nArticle|\n[A-Z]{2,}|\Z)',
    
    # Clause sections: Clause 1.1, Clause 2.3
    r'(Clause\s+\d+\.\d+(?:\.\d+)?)[:\s]*([^.]*?)(?=\nClause|\n[A-Z]{2,}|\Z)',
]

# Section reference patterns in review comments
SECTION_REFERENCE_PATTERNS = [
    r'(\d+\.\d+(?:\.\d+)?)',  # Standard: 1.1, 2.3.4
    r'(Section\s+\d+\.\d+(?:\.\d+)?)',  # With prefix: Section 1.1
    r'(Article\s+[IVX]+|\d+)',  # Articles: Article I, Article 1
    r'(Clause\s+\d+\.\d+(?:\.\d+)?)',  # Clauses: Clause 1.1
]

# Text processing settings
TEXT_PROCESSING = {
    'max_section_length': 2000,  # Maximum characters per section
    'max_context_sections': 5,   # Maximum sections to include in context
    'remove_stopwords': True,
    'apply_stemming': False,
    'apply_lemmatization': True,
}

# System prompts for different types of contracts
SYSTEM_PROMPTS = {
    'default': """You are a legal contract review expert. Your task is to analyze contract sections and provide detailed, professional review comments. Focus on identifying potential risks, ambiguities, and areas that need clarification or modification.""",
    
    'purchase_agreement': """You are a legal expert specializing in purchase agreements. Analyze the contract sections for potential risks, payment terms, delivery conditions, and warranty clauses. Provide detailed professional review comments.""",
    
    'service_agreement': """You are a legal expert specializing in service agreements. Review the contract sections focusing on service levels, performance metrics, liability limitations, and termination clauses. Provide comprehensive professional analysis.""",
    
    'employment_contract': """You are a legal expert specializing in employment contracts. Examine the contract sections for compliance with labor laws, compensation terms, confidentiality clauses, and termination procedures. Provide detailed legal review.""",
    
    'nda': """You are a legal expert specializing in non-disclosure agreements. Analyze the contract sections for scope of confidentiality, permitted disclosures, duration of obligations, and enforcement mechanisms. Provide thorough legal analysis.""",
}

# File processing settings
FILE_PROCESSING = {
    'supported_pdf_extensions': ['.pdf'],
    'supported_word_extensions': ['.docx', '.doc'],
    'encoding': 'utf-8',
    'min_pdfs_per_project': 2,
    'min_word_docs_per_project': 1,
}

# Output formatting
OUTPUT_FORMAT = {
    'jsonl_filename': 'fireworks_training_dataset.jsonl',
    'json_filename': 'fireworks_training_dataset.json',
    'metadata_suffix': '_metadata.json',
    'include_metadata': True,
    'pretty_print_json': True,
}

# Logging configuration
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file_output': False,
    'log_filename': 'dataset_processor.log',
}

# Advanced processing options
ADVANCED_OPTIONS = {
    'use_ocr_fallback': False,  # Use OCR for scanned PDFs
    'parallel_processing': False,  # Process projects in parallel
    'max_workers': 4,  # Number of parallel workers
    'chunk_large_sections': True,  # Split very large sections
    'chunk_size': 1500,  # Characters per chunk
    'overlap_size': 200,  # Overlap between chunks
}

# Quality control settings
QUALITY_CONTROL = {
    'min_comment_length': 10,  # Minimum characters in review comment
    'max_comment_length': 5000,  # Maximum characters in review comment
    'min_section_length': 20,  # Minimum characters in contract section
    'filter_empty_sections': True,
    'validate_section_references': True,
}

# Contract type detection patterns
CONTRACT_TYPE_PATTERNS = {
    'purchase_agreement': [
        r'purchase\s+agreement',
        r'sales?\s+agreement',
        r'buy.*sell',
        r'purchase\s+order',
    ],
    'service_agreement': [
        r'service\s+agreement',
        r'consulting\s+agreement',
        r'professional\s+services',
        r'statement\s+of\s+work',
    ],
    'employment_contract': [
        r'employment\s+agreement',
        r'employment\s+contract',
        r'offer\s+letter',
        r'job\s+agreement',
    ],
    'nda': [
        r'non.?disclosure\s+agreement',
        r'confidentiality\s+agreement',
        r'nda',
        r'secrecy\s+agreement',
    ],
}

def get_system_prompt(contract_type: str = 'default') -> str:
    """Get system prompt based on contract type"""
    return SYSTEM_PROMPTS.get(contract_type.lower(), SYSTEM_PROMPTS['default'])

def detect_contract_type(text: str) -> str:
    """Detect contract type from text content"""
    import re
    
    text_lower = text.lower()
    
    for contract_type, patterns in CONTRACT_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return contract_type
    
    return 'default' 