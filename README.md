# Citation Hallucination Detection

A robust hybrid pipeline for detecting hallucinated citations in academic papers and research documents. The system combines exact bibliographic lookup, fuzzy matching, and optional LLM verification to classify citations as valid, partially valid, or hallucinated.

## Overview

Citation hallucination occurs when AI systems or automated tools generate plausible-looking but non-existent academic references. This pipeline provides a three-stage detection system:

1. **Exact Lookup** (Stage 1): High-precision matching against multiple academic databases
2. **Fuzzy Retrieval** (Stage 2): Semantic similarity and BM25-based candidate retrieval  
3. **LLM Verification** (Stage 3): Optional AI-powered disambiguation and validation

## Features

- **Multi-Source Bibliographic Search**: Integrates with Crossref, OpenAlex, and Semantic Scholar APIs
- **Intelligent Matching**: Combines fuzzy string matching, author similarity (Jaccard), and temporal proximity
- **Flexible LLM Integration**: Supports both OpenAI API and OpenRouter with automatic model detection
- **Robust Error Handling**: Graceful fallbacks and comprehensive debugging information
- **Environment Configuration**: Automatic `.env` loading for API keys and settings
- **Configurable Thresholds**: Tunable confidence thresholds for different use cases

## Classification Labels

- **`valid`**: Citation matches an existing publication with consistent metadata
- **`partially_valid`**: Same paper found but with minor metadata discrepancies (typos, formatting differences)
- **`hallucinated`**: No matching publication found in any database

## Installation

### Prerequisites

- Python 3.8+
- API access to academic databases (free tier available)
- Optional: OpenAI API key or OpenRouter API key for LLM verification

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Vikranth3140/Citation-Hallucination-Detection.git
   cd Citation-Hallucination-Detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** (create `.env` file):
   ```env
   # For LLM verification (choose one)
   OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key-here
   # OR
   OPENAI_API_KEY=sk-your-openai-key-here
   
   # Optional: Custom OpenRouter base URL
   OPENROUTER_BASE=https://openrouter.ai/v1
   ```

## Usage

### Basic Usage

```bash
# Run without LLM verification (faster, uses fuzzy matching only)
python main.py examples.jsonl

# Run with LLM verification (more accurate, requires API key)
python main.py examples.jsonl --llm
```

### Input Format

Create a JSONL file where each line contains a citation to verify:

```jsonl
{"author": "Vaswani A.; Shazeer N.; Parmar N.", "title": "Attention Is All You Need", "year": 2017, "venue": "NeurIPS"}
{"author": "Smith J.", "title": "A Nonexistent Paper About Transformers", "year": 2021, "venue": "Nature"}
```

**Required fields**:
- `author`: Author names (semicolon or comma separated)
- `title`: Paper title

**Optional fields**:
- `year`: Publication year (integer)
- `venue`: Journal or conference name

### Output Format

The pipeline generates a `.verdicts.jsonl` file with verification results:

```jsonl
{
  "author": "Vaswani A.; Shazeer N.; Parmar N.",
  "title": "Attention Is All You Need", 
  "year": 2017,
  "venue": "NeurIPS",
  "label": "valid",
  "confidence": 0.95,
  "matched_source": "semanticscholar",
  "matched_id": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
  "debug": {"exact": {...}}
}
```

**Output fields**:
- `label`: Classification result (`valid`, `partially_valid`, `hallucinated`)
- `confidence`: Confidence score (0.0-1.0)
- `matched_source`: Database that provided the match (`crossref`, `openalex`, `semanticscholar`)
- `matched_id`: Unique identifier from the matched database (DOI, paper ID, etc.)
- `debug`: Detailed information about the matching process

## Configuration

### Model Selection

The pipeline supports multiple LLM providers:

```python
# OpenRouter (default) - Free tier available
detector = Detector(enable_llm=True, openai_model="https://openrouter.ai/mistralai/mistral-7b-instruct:free")

# OpenAI GPT models
detector = Detector(enable_llm=True, openai_model="gpt-4o-mini")

# Other OpenRouter models
detector = Detector(enable_llm=True, openai_model="anthropic/claude-3-haiku")
```

### Confidence Thresholds

Adjust matching thresholds in the code:

```python
class Detector:
    # Stage 1: Exact lookup threshold (default: 0.92)
    if best and best_score >= 0.92:  # High precision
        
    # Stage 2: Fuzzy matching threshold (default: 0.70) 
    return [r for r in ranked if r["_fuzzy"] >= 0.70][:5]
```

### Scoring Weights

Customize the relative importance of different matching factors:

```python
# Exact lookup scoring
score = 0.6*title_score + 0.3*auth_score + 0.1*year_match

# Fuzzy matching scoring  
agg = 0.5*title_score + 0.3*auth_score + 0.2*year_close
```

## Examples

### Example 1: Valid Citation
```bash
# Input
{"author": "LeCun Y.; Bengio Y.; Hinton G.", "title": "Deep learning", "year": 2015, "venue": "Nature"}

# Output  
{
  "label": "valid",
  "confidence": 0.95,
  "matched_source": "crossref", 
  "matched_id": "10.1038/nature14539"
}
```

### Example 2: Partially Valid Citation
```bash
# Input (note typo in author name)
{"author": "Vaswani A.; Shazeer N.; Parmer N.", "title": "Attention Is All You Need", "year": 2017}

# Output
{
  "label": "partially_valid", 
  "confidence": 0.87,
  "matched_source": "semanticscholar",
  "matched_id": "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
}
```

### Example 3: Hallucinated Citation
```bash
# Input
{"author": "Smith J.", "title": "Quantum Machine Learning with Transformers", "year": 2023, "venue": "Science"}

# Output
{
  "label": "hallucinated",
  "confidence": 0.9,
  "matched_source": null,
  "matched_id": null,
  "debug": {"reason": "no_candidates"}
}
```

## Architecture

### Stage 1: Exact Lookup
- Queries Crossref, OpenAlex, and Semantic Scholar APIs
- Computes weighted similarity score:
  - Title similarity (60% weight): Token-based fuzzy matching
  - Author similarity (30% weight): Jaccard similarity on normalized names
  - Year match (10% weight): Exact year matching
- High threshold (≥0.92) ensures precision

### Stage 2: Fuzzy Retrieval  
- Expands search to broader candidate pool
- Uses BM25 ranking on combined title+author+venue text
- Applies fuzzy scoring with temporal proximity bonus
- Returns top 5 candidates above threshold (≥0.70)

### Stage 3: LLM Verification
- Sends structured prompt with candidate metadata to LLM
- Uses strict JSON schema enforcement:
  ```json
  {
    "label": "valid|partially_valid|hallucinated",
    "confidence": 0.85,
    "chosen_index": 2
  }
  ```
- Robust error handling with fallback to fuzzy scores

## API Integration

### Supported Databases

| Database | API Endpoint | Coverage | Rate Limits |
|----------|--------------|----------|-------------|
| **Crossref** | `https://api.crossref.org/works` | 130M+ records | 50 req/sec |
| **OpenAlex** | `https://api.openalex.org/works` | 250M+ works | 100K req/day |  
| **Semantic Scholar** | `https://api.semanticscholar.org` | 200M+ papers | 100 req/sec |

### Error Handling

The pipeline includes comprehensive error handling:

- **Network timeouts**: 20-60 second timeouts with graceful degradation
- **API rate limits**: Automatic retry logic (implement as needed)
- **Malformed responses**: Robust JSON parsing with fallbacks
- **Missing fields**: Default values and optional field handling

## Testing

### Run the pipeline on sample data:

```bash
# Test without LLM (fast)
python main.py examples.jsonl

# Test with LLM verification  
python main.py examples.jsonl --llm

# Check output
cat examples.verdicts.jsonl
```

### Validate configuration:

```bash
# Test environment setup
python -c "import os; print('OPENROUTER_API_KEY:', bool(os.getenv('OPENROUTER_API_KEY')))"

# Test dependencies
python -c "from rapidfuzz import fuzz; from rank_bm25 import BM25Okapi; print('Dependencies OK')"
```

### Optimization Tips

- **Disable LLM for large batches**: Use `--llm` selectively for ambiguous cases
- **Batch processing**: Process citations in chunks for better error recovery
- **Caching**: Consider caching API responses for repeated queries
- **Parallel processing**: Implement multiprocessing for large datasets

## License

This project is licensed under the [MIT License](LICENSE).
