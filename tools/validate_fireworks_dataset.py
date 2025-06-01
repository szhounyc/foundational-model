#!/usr/bin/env python3
"""
Validation script for Fireworks.ai format dataset
"""

import json
import sys
from pathlib import Path

def validate_fireworks_format(jsonl_file):
    """Validate that the dataset follows Fireworks.ai chat format"""
    
    print(f"🔍 Validating Fireworks.ai dataset: {jsonl_file}")
    print("=" * 60)
    
    errors = []
    valid_examples = 0
    total_examples = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_examples += 1
            
            try:
                example = json.loads(line.strip())
                
                # Check required structure
                if 'messages' not in example:
                    errors.append(f"Line {line_num}: Missing 'messages' field")
                    continue
                
                messages = example['messages']
                
                # Check message count (should be 3: system, user, assistant)
                if len(messages) != 3:
                    errors.append(f"Line {line_num}: Expected 3 messages, got {len(messages)}")
                    continue
                
                # Check message roles
                expected_roles = ['system', 'user', 'assistant']
                actual_roles = [msg.get('role') for msg in messages]
                
                if actual_roles != expected_roles:
                    errors.append(f"Line {line_num}: Expected roles {expected_roles}, got {actual_roles}")
                    continue
                
                # Check that all messages have content
                for i, msg in enumerate(messages):
                    if 'content' not in msg or not msg['content'].strip():
                        errors.append(f"Line {line_num}: Message {i+1} missing or empty content")
                        continue
                
                valid_examples += 1
                
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON decode error - {e}")
    
    # Print validation results
    print(f"📊 Validation Results:")
    print(f"   Total examples: {total_examples}")
    print(f"   Valid examples: {valid_examples}")
    print(f"   Invalid examples: {total_examples - valid_examples}")
    print(f"   Success rate: {(valid_examples/total_examples)*100:.1f}%")
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   • {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
    else:
        print("\n✅ All examples are valid!")
    
    return valid_examples == total_examples

def analyze_dataset_content(jsonl_file):
    """Analyze the content of the dataset"""
    
    print(f"\n📈 Dataset Content Analysis:")
    print("=" * 60)
    
    projects = set()
    system_prompts = set()
    user_content_lengths = []
    assistant_content_lengths = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            messages = example['messages']
            
            # Extract project info from user message
            user_content = messages[1]['content']
            if 'Mason project' in user_content:
                projects.add('Mason')
            elif 'Radiant project' in user_content:
                projects.add('Radiant')
            elif 'Vesta project' in user_content:
                projects.add('Vesta')
            
            # Collect system prompts
            system_prompts.add(messages[0]['content'])
            
            # Collect content lengths
            user_content_lengths.append(len(messages[1]['content']))
            assistant_content_lengths.append(len(messages[2]['content']))
    
    print(f"   Projects covered: {sorted(projects)}")
    print(f"   Unique system prompts: {len(system_prompts)}")
    print(f"   Average user content length: {sum(user_content_lengths)/len(user_content_lengths):.0f} chars")
    print(f"   Average assistant content length: {sum(assistant_content_lengths)/len(assistant_content_lengths):.0f} chars")
    print(f"   User content range: {min(user_content_lengths)} - {max(user_content_lengths)} chars")
    print(f"   Assistant content range: {min(assistant_content_lengths)} - {max(assistant_content_lengths)} chars")

def show_sample_examples(jsonl_file, num_samples=2):
    """Show sample examples from the dataset"""
    
    print(f"\n📝 Sample Examples:")
    print("=" * 60)
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Show first and last examples
    for i, line_idx in enumerate([0, len(lines)-1]):
        if i >= num_samples:
            break
            
        example = json.loads(lines[line_idx].strip())
        messages = example['messages']
        
        print(f"\nExample {i+1} (Line {line_idx+1}):")
        print(f"System: {messages[0]['content'][:100]}...")
        print(f"User: {messages[1]['content'][:150]}...")
        print(f"Assistant: {messages[2]['content'][:150]}...")

def main():
    dataset_file = Path("mntn_fireworks_dataset/mntn_contracts_fireworks.jsonl")
    
    if not dataset_file.exists():
        print(f"❌ Dataset file not found: {dataset_file}")
        sys.exit(1)
    
    # Validate format
    is_valid = validate_fireworks_format(dataset_file)
    
    # Analyze content
    analyze_dataset_content(dataset_file)
    
    # Show samples
    show_sample_examples(dataset_file)
    
    # Final summary
    print(f"\n🎯 Dataset Summary:")
    print("=" * 60)
    print(f"   File: {dataset_file}")
    print(f"   Size: {dataset_file.stat().st_size / 1024:.1f} KB")
    print(f"   Format: {'✅ Valid' if is_valid else '❌ Invalid'} Fireworks.ai chat format")
    print(f"   Ready for training: {'✅ Yes' if is_valid else '❌ No'}")

if __name__ == "__main__":
    main() 