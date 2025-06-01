from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Feature Engineering Service",
    description="Service for extracting and engineering features from text data for LLM training",
    version="1.0.0"
)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Failed to download some NLTK data")

class TextFeatureRequest(BaseModel):
    text_data: List[str]
    feature_types: List[str] = ["tfidf", "statistical", "linguistic"]
    max_features: int = 1000
    ngram_range: tuple = (1, 2)
    remove_stopwords: bool = True
    apply_stemming: bool = False
    apply_lemmatization: bool = True

class FeatureExtractionResponse(BaseModel):
    features: Dict
    feature_names: List[str]
    metadata: Dict
    processing_time: float

class DatasetFeatureRequest(BaseModel):
    dataset_id: str
    feature_config: Dict
    output_format: str = "json"  # json, csv, parquet

# Global variables
executor = ThreadPoolExecutor(max_workers=4)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str, remove_stopwords: bool = True, 
                   apply_stemming: bool = False, apply_lemmatization: bool = True) -> str:
    """Preprocess text with various cleaning options"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming
    if apply_stemming:
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Apply lemmatization
    if apply_lemmatization:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def extract_statistical_features(texts: List[str]) -> Dict:
    """Extract statistical features from text"""
    features = {}
    
    for i, text in enumerate(texts):
        # Basic statistics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(sent_tokenize(text))
        
        # Average lengths
        avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Readability metrics (simplified)
        syllable_count = sum([len(re.findall(r'[aeiouAEIOU]', word)) for word in text.split()])
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / word_count)) if word_count > 0 else 0
        
        features[f"text_{i}"] = {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "flesch_score": flesch_score,
            "lexical_diversity": len(set(text.split())) / word_count if word_count > 0 else 0
        }
    
    return features

def extract_linguistic_features(texts: List[str]) -> Dict:
    """Extract linguistic features from text"""
    features = {}
    
    for i, text in enumerate(texts):
        tokens = word_tokenize(text.lower())
        
        # POS tagging
        try:
            pos_tags = nltk.pos_tag(tokens)
            pos_counts = {}
            for _, pos in pos_tags:
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
            # Normalize by text length
            total_tokens = len(tokens)
            if total_tokens > 0:
                pos_ratios = {f"pos_{pos}": count / total_tokens for pos, count in pos_counts.items()}
            else:
                pos_ratios = {}
        except:
            pos_ratios = {}
        
        # Named entity recognition (simplified)
        capitalized_words = len([word for word in text.split() if word[0].isupper()]) if text else 0
        
        features[f"text_{i}"] = {
            **pos_ratios,
            "capitalized_ratio": capitalized_words / len(text.split()) if text.split() else 0,
            "punctuation_ratio": len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
        }
    
    return features

def extract_tfidf_features(texts: List[str], max_features: int = 1000, 
                          ngram_range: tuple = (1, 2)) -> tuple:
    """Extract TF-IDF features from text"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix.toarray(), feature_names

def extract_count_features(texts: List[str], max_features: int = 1000,
                          ngram_range: tuple = (1, 2)) -> tuple:
    """Extract count-based features from text"""
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english'
    )
    
    count_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return count_matrix.toarray(), feature_names

def apply_dimensionality_reduction(features: np.ndarray, method: str = "pca", 
                                 n_components: int = 100) -> tuple:
    """Apply dimensionality reduction to features"""
    if method == "pca":
        reducer = PCA(n_components=min(n_components, features.shape[1]))
    elif method == "svd":
        reducer = TruncatedSVD(n_components=min(n_components, features.shape[1]))
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
    reduced_features = reducer.fit_transform(features)
    explained_variance = getattr(reducer, 'explained_variance_ratio_', None)
    
    return reduced_features, explained_variance

@app.get("/")
async def root():
    return {"message": "Feature Engineering Service", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "feature-engineering"}

@app.post("/extract_features")
async def extract_features(request: TextFeatureRequest):
    """Extract features from text data"""
    start_time = datetime.now()
    
    try:
        # Preprocess texts
        processed_texts = [
            preprocess_text(
                text, 
                request.remove_stopwords, 
                request.apply_stemming, 
                request.apply_lemmatization
            ) for text in request.text_data
        ]
        
        all_features = {}
        feature_names = []
        
        # Extract different types of features
        if "statistical" in request.feature_types:
            stat_features = extract_statistical_features(request.text_data)
            all_features["statistical"] = stat_features
        
        if "linguistic" in request.feature_types:
            ling_features = extract_linguistic_features(request.text_data)
            all_features["linguistic"] = ling_features
        
        if "tfidf" in request.feature_types:
            tfidf_matrix, tfidf_names = extract_tfidf_features(
                processed_texts, 
                request.max_features, 
                request.ngram_range
            )
            all_features["tfidf"] = tfidf_matrix.tolist()
            feature_names.extend([f"tfidf_{name}" for name in tfidf_names])
        
        if "count" in request.feature_types:
            count_matrix, count_names = extract_count_features(
                processed_texts,
                request.max_features,
                request.ngram_range
            )
            all_features["count"] = count_matrix.tolist()
            feature_names.extend([f"count_{name}" for name in count_names])
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        metadata = {
            "num_texts": len(request.text_data),
            "feature_types": request.feature_types,
            "preprocessing": {
                "remove_stopwords": request.remove_stopwords,
                "apply_stemming": request.apply_stemming,
                "apply_lemmatization": request.apply_lemmatization
            },
            "parameters": {
                "max_features": request.max_features,
                "ngram_range": request.ngram_range
            }
        }
        
        return FeatureExtractionResponse(
            features=all_features,
            feature_names=feature_names,
            metadata=metadata,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_dataset")
async def process_dataset(request: DatasetFeatureRequest):
    """Process a complete dataset and extract features"""
    try:
        # Load dataset from processed data
        dataset_path = f"/app/data/{request.dataset_id}"
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load text data
        texts = []
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if 'text' in data:
                        texts.append(data['text'])
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'chunks' in data:
                    texts = [chunk['text'] for chunk in data['chunks']]
        
        if not texts:
            raise HTTPException(status_code=400, detail="No text data found in dataset")
        
        # Extract features using the configuration
        feature_request = TextFeatureRequest(
            text_data=texts,
            **request.feature_config
        )
        
        # Process in background
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, extract_features_sync, feature_request)
        
        # Save results
        output_dir = "/app/features"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{request.dataset_id}_features.{request.output_format}")
        
        if request.output_format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        elif request.output_format == "csv":
            # Convert to DataFrame and save as CSV
            df = pd.DataFrame(result["features"])
            df.to_csv(output_file, index=False)
        
        return {
            "status": "success",
            "dataset_id": request.dataset_id,
            "output_file": output_file,
            "num_features": len(result["feature_names"]),
            "processing_time": result["processing_time"]
        }
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_features_sync(request: TextFeatureRequest) -> dict:
    """Synchronous version of feature extraction for background processing"""
    # This is a simplified version - in practice, you'd implement the full logic here
    return {
        "features": {},
        "feature_names": [],
        "processing_time": 0.0
    }

@app.get("/datasets")
async def list_datasets():
    """List available datasets for feature extraction"""
    data_dir = "/app/data"
    if not os.path.exists(data_dir):
        return {"datasets": []}
    
    datasets = []
    for filename in os.listdir(data_dir):
        if filename.endswith(('.json', '.jsonl')):
            file_path = os.path.join(data_dir, filename)
            try:
                stat = os.stat(file_path)
                datasets.append({
                    "id": filename,
                    "name": filename,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                logger.error(f"Error reading dataset {filename}: {e}")
    
    return {"datasets": datasets}

@app.get("/features/{dataset_id}")
async def get_extracted_features(dataset_id: str):
    """Get previously extracted features for a dataset"""
    features_dir = "/app/features"
    feature_files = []
    
    if os.path.exists(features_dir):
        for filename in os.listdir(features_dir):
            if filename.startswith(dataset_id):
                file_path = os.path.join(features_dir, filename)
                try:
                    stat = os.stat(file_path)
                    feature_files.append({
                        "filename": filename,
                        "size_bytes": stat.st_size,
                        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error reading feature file {filename}: {e}")
    
    return {"dataset_id": dataset_id, "feature_files": feature_files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004) 