#!/usr/bin/env python3
"""
Dataset Preparation Script for Fireworks.ai Fine-tuning

This script processes training datasets containing:
- 2 PDF contract files per project
- 1 Word document with review comments per project

The script extracts text, maps section references, and formats data for Fireworks.ai training.
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse

# PDF processing
import PyPDF2
import fitz  # PyMuPDF for better text extraction

# Word document processing
from docx import Document

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProjectData:
    """Data structure for a single project"""
    project_id: str
    pdf_files: List[str]
    word_file: str
    pdf_texts: List[str]
    word_text: str
    sections: Dict[str, str]  # section_number -> section_text
    review_comments: List[Dict]

class ContractSectionExtractor:
    """Extract and map sections from PDF contracts"""
    
    def __init__(self):
        # Common section patterns in contracts
        self.section_patterns = [
            r'(\d+\.\d+(?:\.\d+)?)\s+([A-Z][^.]*?)(?=\n\d+\.\d+|\n[A-Z]{2,}|\Z)',
            r'(Section\s+\d+\.\d+(?:\.\d+)?)[:\s]+([^.]*?)(?=\nSection|\n[A-Z]{2,}|\Z)',
            r'(\d+\.\d+(?:\.\d+)?)[:\s]*([A-Z][^.]*?)(?=\n\d+\.\d+|\n[A-Z]{2,}|\Z)',
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF for better accuracy"""
        try:
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
                if len(section_text) > 2000:  # Limit section length
                    section_text = section_text[:2000] + "..."
                
                sections[section_num] = section_text
        
        return sections

class WordDocumentProcessor:
    """Process Word documents containing review comments"""
    
    def __init__(self):
        self.section_ref_pattern = r'(\d+\.\d+(?:\.\d+)?)'
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from Word document"""
        try:
            doc = Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from {docx_path}: {e}")
            return ""
    
    def parse_review_comments(self, text: str) -> List[Dict]:
        """Parse review comments and extract section references"""
        comments = []
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        current_comment = ""
        for paragraph in paragraphs:
            # Check if this paragraph contains section references
            section_refs = re.findall(self.section_ref_pattern, paragraph)
            
            if section_refs:
                # This paragraph references specific sections
                if current_comment:
                    # Save previous comment
                    comments.append({
                        "comment": current_comment.strip(),
                        "section_references": []
                    })
                
                # Start new comment with section references
                current_comment = paragraph
                comments.append({
                    "comment": paragraph,
                    "section_references": section_refs
                })
                current_comment = ""
            else:
                # Continue building current comment
                current_comment += " " + paragraph
        
        # Add final comment if exists
        if current_comment.strip():
            comments.append({
                "comment": current_comment.strip(),
                "section_references": []
            })
        
        return comments

class FireworksDatasetFormatter:
    """Format data for Fireworks.ai fine-tuning"""
    
    def __init__(self):
        self.system_prompt = """You are a legal contract review expert. Your task is to analyze contract sections and provide detailed, professional review comments. Focus on identifying potential risks, ambiguities, and areas that need clarification or modification."""
    
    def create_training_examples(self, project: ProjectData) -> List[Dict]:
        """Create training examples in Fireworks.ai format"""
        examples = []
        
        # Combine all PDF sections for context
        all_sections = {}
        for i, pdf_text in enumerate(project.pdf_texts):
            extractor = ContractSectionExtractor()
            sections = extractor.extract_sections(pdf_text)
            for sec_num, sec_text in sections.items():
                all_sections[sec_num] = sec_text
        
        # Process each review comment
        for comment_data in project.review_comments:
            comment = comment_data["comment"]
            section_refs = comment_data["section_references"]
            
            if not section_refs:
                # General comment without specific section reference
                context = self._create_general_context(all_sections)
                user_prompt = f"Please review this contract and provide professional comments:\n\n{context}"
            else:
                # Comment referencing specific sections
                context = self._create_section_context(all_sections, section_refs)
                sections_text = ", ".join(section_refs)
                user_prompt = f"Please review section(s) {sections_text} of this contract:\n\n{context}"
            
            # Create training example
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
    
    def _create_section_context(self, sections: Dict[str, str], section_refs: List[str]) -> str:
        """Create context from specific sections"""
        context_parts = []
        
        for section_ref in section_refs:
            if section_ref in sections:
                context_parts.append(f"Section {section_ref}:\n{sections[section_ref]}")
            else:
                # Try to find partial matches
                for sec_num, sec_text in sections.items():
                    if sec_num.startswith(section_ref) or section_ref in sec_num:
                        context_parts.append(f"Section {sec_num}:\n{sec_text}")
                        break
        
        return "\n\n".join(context_parts)
    
    def _create_general_context(self, sections: Dict[str, str], max_sections: int = 5) -> str:
        """Create general context from multiple sections"""
        # Select most relevant sections (first few sections typically)
        sorted_sections = sorted(sections.items(), key=lambda x: self._section_sort_key(x[0]))
        selected_sections = sorted_sections[:max_sections]
        
        context_parts = []
        for sec_num, sec_text in selected_sections:
            context_parts.append(f"Section {sec_num}:\n{sec_text}")
        
        return "\n\n".join(context_parts)
    
    def _section_sort_key(self, section_num: str) -> Tuple:
        """Create sort key for section numbers"""
        try:
            parts = [int(x) for x in section_num.split('.')]
            return tuple(parts)
        except:
            return (999, 999, 999)  # Put unparseable sections at the end

class DatasetProcessor:
    """Main dataset processor"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.section_extractor = ContractSectionExtractor()
        self.word_processor = WordDocumentProcessor()
        self.formatter = FireworksDatasetFormatter()
    
    def discover_projects(self) -> List[str]:
        """Discover project directories in input folder"""
        projects = []
        
        for item in self.input_dir.iterdir():
            if item.is_dir():
                # Check if directory contains required files
                pdf_files = list(item.glob("*.pdf"))
                word_files = list(item.glob("*.docx")) + list(item.glob("*.doc"))
                
                if len(pdf_files) >= 2 and len(word_files) >= 1:
                    projects.append(item.name)
                else:
                    logger.warning(f"Project {item.name} doesn't have required files (2 PDFs + 1 Word doc)")
        
        return projects
    
    def process_project(self, project_id: str) -> Optional[ProjectData]:
        """Process a single project"""
        project_dir = self.input_dir / project_id
        
        # Find files
        pdf_files = sorted(list(project_dir.glob("*.pdf")))[:2]  # Take first 2 PDFs
        word_files = list(project_dir.glob("*.docx")) + list(project_dir.glob("*.doc"))
        
        if len(pdf_files) < 2 or len(word_files) < 1:
            logger.error(f"Project {project_id} missing required files")
            return None
        
        word_file = word_files[0]  # Take first Word document
        
        logger.info(f"Processing project {project_id}")
        logger.info(f"  PDFs: {[f.name for f in pdf_files]}")
        logger.info(f"  Word: {word_file.name}")
        
        # Extract text from PDFs
        pdf_texts = []
        sections = {}
        
        for pdf_file in pdf_files:
            text = self.section_extractor.extract_text_from_pdf(str(pdf_file))
            pdf_texts.append(text)
            
            # Extract sections
            pdf_sections = self.section_extractor.extract_sections(text)
            sections.update(pdf_sections)
        
        # Extract text from Word document
        word_text = self.word_processor.extract_text_from_docx(str(word_file))
        
        # Parse review comments
        review_comments = self.word_processor.parse_review_comments(word_text)
        
        return ProjectData(
            project_id=project_id,
            pdf_files=[str(f) for f in pdf_files],
            word_file=str(word_file),
            pdf_texts=pdf_texts,
            word_text=word_text,
            sections=sections,
            review_comments=review_comments
        )
    
    def process_all_projects(self) -> List[Dict]:
        """Process all projects and create training dataset"""
        projects = self.discover_projects()
        logger.info(f"Found {len(projects)} projects: {projects}")
        
        all_examples = []
        
        for project_id in projects:
            try:
                project_data = self.process_project(project_id)
                if project_data:
                    examples = self.formatter.create_training_examples(project_data)
                    all_examples.extend(examples)
                    logger.info(f"Generated {len(examples)} training examples from {project_id}")
                    
                    # Save project metadata
                    self._save_project_metadata(project_data)
                    
            except Exception as e:
                logger.error(f"Error processing project {project_id}: {e}")
                continue
        
        return all_examples
    
    def _save_project_metadata(self, project: ProjectData):
        """Save project metadata for debugging"""
        metadata = {
            "project_id": project.project_id,
            "pdf_files": project.pdf_files,
            "word_file": project.word_file,
            "sections_found": list(project.sections.keys()),
            "review_comments_count": len(project.review_comments),
            "review_comments": project.review_comments
        }
        
        metadata_file = self.output_dir / f"{project.project_id}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def save_dataset(self, examples: List[Dict], filename: str = "fireworks_training_dataset.jsonl"):
        """Save dataset in Fireworks.ai JSONL format"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} training examples to {output_file}")
        
        # Also save as regular JSON for inspection
        json_file = self.output_dir / filename.replace('.jsonl', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        return str(output_file)

def main():
    parser = argparse.ArgumentParser(description="Process contract datasets for Fireworks.ai fine-tuning")
    parser.add_argument("--input-dir", required=True, help="Input directory containing project folders")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed dataset")
    parser.add_argument("--output-filename", default="fireworks_training_dataset.jsonl", 
                       help="Output filename for the training dataset")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DatasetProcessor(args.input_dir, args.output_dir)
    
    # Process all projects
    examples = processor.process_all_projects()
    
    if examples:
        # Save dataset
        output_file = processor.save_dataset(examples, args.output_filename)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Dataset Processing Complete!")
        print(f"{'='*50}")
        print(f"Total training examples: {len(examples)}")
        print(f"Output file: {output_file}")
        print(f"Format: Fireworks.ai JSONL")
        print(f"\nTo use with Fireworks.ai:")
        print(f"1. Upload the JSONL file to Fireworks.ai")
        print(f"2. Create a fine-tuning job")
        print(f"3. Use the trained model for contract review")
    else:
        print("No training examples generated. Check your input data.")

if __name__ == "__main__":
    main() 