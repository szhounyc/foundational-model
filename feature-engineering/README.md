# Feature Engineering Service

A specialized service for extracting and engineering features from text data to improve LLM training performance.

## Features

- 📊 **Statistical Features**: Extract text statistics like word count, sentence length, readability scores
- 🔤 **Linguistic Features**: POS tagging, named entity recognition, syntactic analysis
- 🔍 **TF-IDF Features**: Term frequency-inverse document frequency vectorization
- 📈 **Count Features**: N-gram count vectorization
- 🎯 **Dimensionality Reduction**: PCA and SVD for feature compression
- 🧹 **Text Preprocessing**: Stopword removal, stemming, lemmatization

## API Endpoints

### Health Check
- `GET /health` - Service health status

### Feature Extraction
- `POST /extract_features` - Extract features from text data
- `POST /process_dataset` - Process complete datasets
- `GET /datasets` - List available datasets
- `GET /features/{dataset_id}` - Get extracted features

## Usage

### Extract Features from Text

```python
import requests

data = {
    "text_data": ["This is sample text.", "Another example text."],
    "feature_types": ["tfidf", "statistical", "linguistic"],
    "max_features": 1000,
    "ngram_range": [1, 2],
    "remove_stopwords": True,
    "apply_lemmatization": True
}

response = requests.post("http://localhost:8004/extract_features", json=data)
features = response.json()
```

### Process Complete Dataset

```python
data = {
    "dataset_id": "my_dataset.jsonl",
    "feature_config": {
        "feature_types": ["tfidf", "statistical"],
        "max_features": 500
    },
    "output_format": "json"
}

response = requests.post("http://localhost:8004/process_dataset", json=data)
```

## Feature Types

### Statistical Features
- Character count
- Word count
- Sentence count
- Average word length
- Average sentence length
- Flesch readability score
- Lexical diversity

### Linguistic Features
- Part-of-speech tag ratios
- Capitalized word ratio
- Punctuation ratio
- Named entity density

### TF-IDF Features
- Term frequency-inverse document frequency
- Configurable n-gram ranges
- Stopword filtering
- Maximum feature limits

### Count Features
- Raw term counts
- N-gram counts
- Binary occurrence features

## Configuration

The service can be configured through environment variables:

- `PYTHONPATH=/app` - Python path configuration
- Port: `8004`

## Dependencies

- FastAPI for API framework
- scikit-learn for feature extraction
- NLTK for natural language processing
- NumPy/Pandas for data manipulation
- Pydantic for data validation

## Integration

The feature engineering service integrates with:
- PDF Processor: Processes extracted text
- Model Trainer: Provides enhanced features for training
- Backend API: Coordinates feature extraction workflows

## Output Formats

- JSON: Structured feature data
- CSV: Tabular format for analysis
- Parquet: Compressed columnar format (planned)

## Performance

- Asynchronous processing for large datasets
- Background task execution
- Memory-efficient feature extraction
- Configurable batch processing 