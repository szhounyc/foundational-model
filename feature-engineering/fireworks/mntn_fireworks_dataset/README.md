# MNTN Contracts Dataset - Fireworks.ai Format

This dataset contains real estate contract review examples extracted from Manhattan (MNTN) contract projects, formatted for Fireworks.ai fine-tuning.

## Dataset Overview

- **Total Examples**: 89 training examples
- **Format**: Fireworks.ai chat format (JSONL)
- **Domain**: Real estate contract review and legal analysis
- **Projects**: 3 Manhattan real estate projects (Mason, Radiant, Vesta)
- **Size**: 103.2 KB

## Dataset Structure

Each example follows the Fireworks.ai chat format with three messages:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert legal contract reviewer..."
    },
    {
      "role": "user", 
      "content": "Please provide a general review of this contract..."
    },
    {
      "role": "assistant",
      "content": "Professional legal review comment..."
    }
  ]
}
```

## Projects Included

### 1. Mason Project
- **Files**: 2 PDFs + 1 Word document with review comments
- **Location**: 40-46 24th Street, Unit 5G
- **Documents**: Purchase Agreement, Rider, Contract Comments

### 2. Radiant Project  
- **Files**: 2 PDFs + 1 Word document with review comments
- **Location**: 24-01 Queens Plaza North, Unit 802
- **Documents**: Purchase Agreement, Rider, Contract Comments

### 3. Vesta Project
- **Files**: 2 PDFs + 1 Word document with review comments
- **Location**: Vesta Condominium, Unit 303A
- **Documents**: Purchase Agreement, Rider, Contract Comments

## Content Statistics

- **Average user content length**: 511 characters
- **Average assistant content length**: 206 characters
- **User content range**: 457 - 627 characters
- **Assistant content range**: 32 - 1,224 characters
- **Estimated tokens**: ~26,437 tokens

## System Prompt

The dataset uses a consistent system prompt that defines the AI's role:

```
You are an expert legal contract reviewer specializing in real estate transactions. Your role is to analyze contract sections and provide detailed, professional review comments that identify potential risks, ambiguities, and areas requiring clarification or modification. Focus on practical legal concerns that could affect the parties involved.
```

## Training Examples Cover

- **421A tax abatement requirements and implications**
- **Certificate of occupancy status and requirements**
- **Offering plan effectiveness and compliance**
- **Unit sales and leasing status**
- **Construction status and closing timelines**
- **Financial statement requirements**
- **Appliance and fixture inclusions**
- **Building maintenance and repair history**
- **Financing arrangements and CEMA requirements**
- **Pet policies and restrictions**
- **Lease term restrictions and requirements**
- **Insurance requirements**
- **Mortgage recording tax credits**
- **Sponsor legal fees and ACRIS fees**
- **Closing date negotiations**
- **Termination clauses and deposit refunds**
- **Punchlist item timelines**
- **Lease restriction modifications**

## Files

- `mntn_contracts_fireworks.jsonl` - Main training dataset
- `dataset_summary.json` - Dataset statistics and metadata
- `README.md` - This documentation

## Usage

### Validation
```bash
python validate_fireworks_dataset.py
```

### Training
```bash
python train_mntn_contracts.py
```

### Direct Training with Enhanced Trainer
```bash
cd model-trainer
python main.py  # Start the training service

# In another terminal:
python train_mntn_contracts.py
```

## Training Configuration

Recommended training parameters for this dataset:

```json
{
  "model_name": "microsoft/DialoGPT-small",
  "num_epochs": 3,
  "batch_size": 4,
  "learning_rate": 5e-5,
  "max_length": 512,
  "use_lora": true,
  "lora_config": {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["c_attn", "c_proj"],
    "lora_dropout": 0.1
  }
}
```

## Quality Assurance

- ✅ All 89 examples validated for Fireworks.ai format compliance
- ✅ Consistent system prompt across all examples
- ✅ Real-world contract review scenarios
- ✅ Professional legal language and terminology
- ✅ Diverse contract sections and review types

## Expected Model Capabilities

After training on this dataset, the model should be able to:

1. **Analyze contract sections** for potential legal issues
2. **Identify risks and ambiguities** in real estate agreements
3. **Provide professional review comments** in legal terminology
4. **Ask relevant questions** about contract terms and conditions
5. **Suggest modifications** to protect client interests
6. **Understand real estate transaction processes** and requirements

## Limitations

- Dataset size is relatively small (89 examples)
- Focused specifically on Manhattan real estate transactions
- May require additional data for broader real estate markets
- Professional legal review should always be conducted by qualified attorneys

## License and Usage

This dataset is derived from real contract review work and should be used responsibly for training AI models to assist in legal document review. The trained models should supplement, not replace, professional legal advice.

## Generated By

- **Processor**: `dataset_processor_fireworks.py`
- **Source**: Manhattan real estate contract projects
- **Format**: Fireworks.ai chat completion format
- **Date**: Generated from MNTN project documents 