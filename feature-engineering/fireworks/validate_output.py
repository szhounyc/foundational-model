#!/usr/bin/env python3
import json

def validate_jsonl(file_path):
    valid_count = 0
    total_count = 0
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                total_count += 1
                try:
                    data = json.loads(line)
                    assert 'messages' in data
                    assert len(data['messages']) == 3
                    roles = [msg['role'] for msg in data['messages']]
                    assert roles == ['system', 'user', 'assistant']
                    for msg in data['messages']:
                        assert 'content' in msg
                        assert msg['content'].strip()
                    valid_count += 1
                except Exception as e:
                    print(f'Line {line_num}: {e}')
    
    print(f'Validation Results:')
    print(f'Total examples: {total_count}')
    print(f'Valid examples: {valid_count}')
    print(f'Format compliance: {valid_count/total_count*100:.1f}%')
    return valid_count == total_count

if __name__ == "__main__":
    validate_jsonl('processed_datasets/mntn_contracts_training.jsonl') 