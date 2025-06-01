#!/usr/bin/env python3
"""
MNTN Contract Dataset Processor for Fireworks.ai Format

This script processes the MNTN contract dataset containing:
- 3 projects (Mason, Radiant, Vesta)
- Each project has 2 PDF contracts and 1 Word document with review comments
- Converts to Fireworks.ai chat format for fine-tuning
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# PDF processing
try:
    import PyPDF2
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PDF processing libraries not available. Install PyPDF2 and PyMuPDF.")

# Word document processing
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("Warning: python-docx not available. Install python-docx.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContractProcessor:
    """Process contract files and extract structured data"""
    
    def __init__(self):
        self.section_patterns = [
            r'(\d+\.\d+(?:\.\d+)?)\s*[:\-\s]*([A-Z][^.]*?)(?=\n\d+\.\d+|\n[A-Z]{2,}|\Z)',
            r'(Section\s+\d+\.\d+(?:\.\d+)?)[:\s]*([^.]*?)(?=\nSection|\n[A-Z]{2,}|\Z)',
            r'(\d+\.\d+(?:\.\d+)?)[:\s]*([A-Z][^.]*?)(?=\n\d+\.\d+|\n[A-Z]{2,}|\Z)',
        ]
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            logger.error("PDF processing libraries not available")
            return ""
        
        try:
            # Try PyMuPDF first (better text extraction)
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {pdf_path}, trying PyPDF2: {e}")
            return self._extract_with_pypdf2(pdf_path)
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Fallback PDF extraction using PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""
    
    def extract_docx_text(self, docx_path: str) -> str:
        """Extract text from Word document"""
        if not DOCX_AVAILABLE:
            logger.error("python-docx not available")
            return ""
        
        try:
            doc = Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from {docx_path}: {e}")
            return ""
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract numbered sections from contract text"""
        sections = {}
        
        for pattern in self.section_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                section_num = match.group(1).strip()
                section_text = match.group(2).strip()
                
                # Clean section number
                section_num = re.sub(r'^Section\s+', '', section_num, flags=re.IGNORECASE)
                
                # Clean and limit section text
                section_text = re.sub(r'\s+', ' ', section_text)
                if len(section_text) > 1500:  # Limit section length
                    section_text = section_text[:1500] + "..."
                
                if len(section_text) > 50:  # Only keep substantial sections
                    sections[section_num] = section_text
        
        return sections
    
    def parse_review_comments(self, text: str) -> List[Dict]:
        """Parse review comments from Word document"""
        comments = []
        
        # Split into paragraphs and clean
        paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 10]
        
        section_ref_pattern = r'(\d+\.\d+(?:\.\d+)?)'
        
        for paragraph in paragraphs:
            # Find section references in this paragraph
            section_refs = re.findall(section_ref_pattern, paragraph)
            
            # Clean the comment text
            clean_comment = re.sub(r'\s+', ' ', paragraph).strip()
            
            if len(clean_comment) > 30:  # Only keep substantial comments
                comments.append({
                    "comment": clean_comment,
                    "section_references": section_refs,
                    "has_section_refs": len(section_refs) > 0
                })
        
        return comments

class FireworksFormatter:
    """Format data for Fireworks.ai training"""
    
    def __init__(self):
        self.system_prompt = """You are an expert legal contract reviewer specializing in real estate transactions. Your role is to analyze contract sections and provide detailed, professional review comments that identify potential risks, ambiguities, and areas requiring clarification or modification. Focus on practical legal concerns that could affect the parties involved."""
    
    def create_training_examples(self, project_name: str, pdf_sections: Dict[str, str], 
                               review_comments: List[Dict]) -> List[Dict]:
        """Create training examples in Fireworks.ai chat format"""
        examples = []
        
        for comment_data in review_comments:
            comment = comment_data["comment"]
            section_refs = comment_data["section_references"]
            
            if comment_data["has_section_refs"] and section_refs:
                # Create example with specific section context
                context_sections = []
                for ref in section_refs:
                    if ref in pdf_sections:
                        context_sections.append(f"Section {ref}: {pdf_sections[ref]}")
                
                if context_sections:
                    context = "\n\n".join(context_sections)
                    user_prompt = f"Please review the following contract section(s) from the {project_name} project:\n\n{context}"
                    
                    example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": self.system_prompt
                            },
                            {
                                "role": "user", 
                                "content": user_prompt
                            },
                            {
                                "role": "assistant",
                                "content": comment
                            }
                        ]
                    }
                    examples.append(example)
            
            else:
                # General comment - use broader context
                if len(pdf_sections) > 0:
                    # Take first few sections as general context
                    context_sections = list(pdf_sections.items())[:3]
                    context = "\n\n".join([f"Section {k}: {v}" for k, v in context_sections])
                    
                    user_prompt = f"Please provide a general review of this contract from the {project_name} project:\n\n{context}"
                    
                    example = {
                        "messages": [
                            {
                                "role": "system", 
                                "content": self.system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            },
                            {
                                "role": "assistant",
                                "content": comment
                            }
                        ]
                    }
                    examples.append(example)
        
        return examples

class MNTNDatasetProcessor:
    """Main processor for MNTN contract dataset"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.processor = ContractProcessor()
        self.formatter = FireworksFormatter()
        
    def process_project(self, project_name: str) -> List[Dict]:
        """Process a single project and return training examples"""
        project_path = self.dataset_path / project_name
        
        if not project_path.exists():
            logger.error(f"Project path does not exist: {project_path}")
            return []
        
        logger.info(f"Processing project: {project_name}")
        
        # Find files
        pdf_files = list(project_path.glob("*.pdf"))
        docx_files = list(project_path.glob("*.docx"))
        
        if len(pdf_files) != 2:
            logger.warning(f"Expected 2 PDF files, found {len(pdf_files)} in {project_name}")
        
        if len(docx_files) != 1:
            logger.warning(f"Expected 1 DOCX file, found {len(docx_files)} in {project_name}")
            return []
        
        # Extract text from PDFs
        all_sections = {}
        for pdf_file in pdf_files:
            logger.info(f"Processing PDF: {pdf_file.name}")
            pdf_text = self.processor.extract_pdf_text(str(pdf_file))
            sections = self.processor.extract_sections(pdf_text)
            all_sections.update(sections)
        
        logger.info(f"Extracted {len(all_sections)} sections from PDFs")
        
        # Extract review comments from Word document
        docx_file = docx_files[0]
        logger.info(f"Processing DOCX: {docx_file.name}")
        docx_text = self.processor.extract_docx_text(str(docx_file))
        review_comments = self.processor.parse_review_comments(docx_text)
        
        logger.info(f"Extracted {len(review_comments)} review comments")
        
        # Create training examples
        examples = self.formatter.create_training_examples(
            project_name, all_sections, review_comments
        )
        
        logger.info(f"Created {len(examples)} training examples for {project_name}")
        return examples
    
    def process_all_projects(self) -> List[Dict]:
        """Process all projects and return combined training examples"""
        all_examples = []
        
        # Process each project
        projects = ["Mason", "Radiant", "Vesta"]
        
        for project in projects:
            examples = self.process_project(project)
            all_examples.extend(examples)
        
        return all_examples
    
    def save_dataset(self, examples: List[Dict], output_path: str):
        """Save training examples in JSONL format"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} training examples to {output_path}")
    
    def create_summary(self, examples: List[Dict], output_dir: str):
        """Create a summary of the processed dataset"""
        summary = {
            "total_examples": len(examples),
            "projects_processed": ["Mason", "Radiant", "Vesta"],
            "format": "fireworks_ai_chat",
            "system_prompt": self.formatter.system_prompt,
            "example_structure": {
                "messages": [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "Contract section(s) to review"},
                    {"role": "assistant", "content": "Professional review comment"}
                ]
            }
        }
        
        # Add statistics
        total_chars = sum(len(json.dumps(ex)) for ex in examples)
        avg_chars = total_chars / len(examples) if examples else 0
        
        summary["statistics"] = {
            "average_example_length": avg_chars,
            "total_dataset_size": total_chars,
            "estimated_tokens": total_chars // 4  # Rough estimate
        }
        
        summary_path = Path(output_dir) / "dataset_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset summary saved to {summary_path}")
        return summary

def main():
    """Main function to process the MNTN dataset"""
    # Configuration
    dataset_path = "/Users/sz/src/project/genai/agent/semantic-search/foundational-model/dataset/zlg-re/MNTN"
    output_dir = "mntn_fireworks_dataset"
    output_file = f"{output_dir}/mntn_contracts_fireworks.jsonl"
    
    # Create processor
    processor = MNTNDatasetProcessor(dataset_path)
    
    # Process all projects
    logger.info("Starting MNTN dataset processing...")
    examples = processor.process_all_projects()
    
    if not examples:
        logger.error("No training examples were created!")
        return
    
    # Save dataset
    processor.save_dataset(examples, output_file)
    
    # Create summary
    summary = processor.create_summary(examples, output_dir)
    
    # Print results
    print("\n" + "="*60)
    print("MNTN DATASET PROCESSING COMPLETE")
    print("="*60)
    print(f"Total training examples: {summary['total_examples']}")
    print(f"Projects processed: {', '.join(summary['projects_processed'])}")
    print(f"Output file: {output_file}")
    print(f"Average example length: {summary['statistics']['average_example_length']:.0f} characters")
    print(f"Estimated tokens: {summary['statistics']['estimated_tokens']:,}")
    print("\nDataset is ready for Fireworks.ai fine-tuning!")
    print("="*60)

if __name__ == "__main__":
    main() 