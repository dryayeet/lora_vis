# Real-World Tasks Guide

This document explains the different task types available in the Knowledge Change Simulator and how to interpret their outputs.

## Overview

The Knowledge Change Simulator supports four task types, each designed to demonstrate how LoRA fine-tuning can be applied to real-world NLP problems:

1. **Sentiment Analysis** - Binary classification (positive/negative)
2. **Text Classification** - Multi-class categorization
3. **Text Similarity** - Embedding generation for similarity search
4. **Generic Regression** - Continuous value prediction

---

## 1. 😊 Sentiment Analysis

### What It Does
Classifies text as having positive or negative sentiment. This is a fundamental NLP task used in many real-world applications.

### Output Format
**2-dimensional probability vector**: `[negative_probability, positive_probability]`

Example outputs:
- `[0.8, 0.2]` → 80% negative, 20% positive → **Prediction: Negative**
- `[0.1, 0.9]` → 10% negative, 90% positive → **Prediction: Positive**
- `[0.5, 0.5]` → Neutral/uncertain

### How to Use
1. Select **"Sentiment Analysis (Positive/Negative)"** from the task dropdown
2. Enter training samples with clear positive or negative sentiment
3. Example training data:
   ```
   I love this amazing product!
   This is terrible and awful
   Great service, very happy
   Hate this worst experience
   ```
4. After training, the output shows:
   - **Before Training**: Probability bars for negative/positive
   - **After Training**: Updated probabilities showing learning
   - **Prediction**: The class with highest probability

### Real-World Applications

**Customer Review Analysis**
- Automatically categorize product reviews as positive/negative
- Track customer satisfaction trends
- Identify critical negative feedback

**Social Media Monitoring**
- Monitor brand sentiment on social platforms
- Detect negative mentions early
- Measure campaign effectiveness

**Feedback Classification**
- Route negative feedback to support teams
- Identify satisfied customers for testimonials
- Aggregate sentiment scores for reporting

### How It Works

The model learns to map text features to sentiment probabilities:
- **Input**: 8D text feature vector (extracted from text using TF-IDF-like features)
- **Process**: Text → MLP → Softmax activation
- **Output**: 2D probability vector (sums to 1.0)

**Training Process:**
1. Text is encoded into numerical features (length, word counts, sentiment keywords)
2. Labels are inferred from keywords (e.g., "love", "amazing" → positive; "hate", "terrible" → negative)
3. Model learns to predict these labels through LoRA fine-tuning
4. Output probabilities reflect model confidence

---

## 2. 📂 Text Classification

### What It Does
Categorizes text into predefined categories: **News**, **Review**, **Question**, or **Comment**. Useful for organizing and routing documents.

### Output Format
**4-dimensional probability vector**: `[News, Review, Question, Comment]`

Example outputs:
- `[0.8, 0.1, 0.05, 0.05]` → **Prediction: News**
- `[0.1, 0.85, 0.03, 0.02]` → **Prediction: Review**
- `[0.05, 0.1, 0.8, 0.05]` → **Prediction: Question**
- `[0.2, 0.2, 0.1, 0.5]` → **Prediction: Comment**

### How to Use
1. Select **"Text Classification (Categories)"** from the task dropdown
2. Enter diverse training samples across different categories
3. Example training data:
   ```
   What is the weather today?
   I reviewed the new smartphone and it's excellent
   Breaking news: Scientists discover new planet
   This movie was fantastic!
   ```
4. After training, view:
   - **Before/After Comparison**: Side-by-side probability bar charts
   - **Prediction**: The category with highest probability

### Real-World Applications

**Email Routing**
- Automatically route emails to appropriate departments
- Classify: Support questions → Help desk, News → Newsletters

**Content Management**
- Organize articles, blog posts, documents by type
- Filter content for different audiences
- Build content recommendation systems

**Customer Service**
- Classify support tickets by type
- Route questions to FAQs vs. human agents
- Track issue categories for analytics

### Category Detection Logic

The system uses pattern matching to infer labels:
- **Questions**: Contains "?", "what", "how", "why", "when", "where"
- **Reviews**: Contains "review", "rating", "star", "bought", "product"
- **News**: Contains "news", "report", "announced", "breaking"
- **Comments**: Default category for general text

During fine-tuning, the model learns more nuanced patterns beyond keyword matching.

---

## 3. 🔍 Text Similarity

### What It Does
Generates **embedding vectors** that capture semantic meaning. Similar texts produce similar embeddings, enabling similarity search and comparison.

### Output Format
**8-dimensional embedding vector**: `[v1, v2, v3, v4, v5, v6, v7, v8]`

Example outputs:
- Text A: `[0.2, -0.5, 0.8, 0.1, -0.3, 0.6, -0.2, 0.4]`
- Text B (similar): `[0.25, -0.48, 0.82, 0.12, -0.28, 0.58, -0.18, 0.42]` → **High similarity**
- Text C (different): `[-0.3, 0.6, -0.2, 0.5, 0.1, -0.4, 0.7, -0.1]` → **Low similarity**

### How to Use
1. Select **"Text Similarity (Embeddings)"** from the task dropdown
2. Enter text samples (including similar pairs)
3. Example training data:
   ```
   The cat sat on the mat
   The dog played in the yard
   Birds fly in the sky
   A feline rests on a rug  (similar to first line)
   ```
4. After training, see:
   - **Embedding Vectors**: Numerical representation of text
   - **Cosine Similarity**: Measure of how similar embeddings are (0.0 to 1.0)
   - **Before/After Comparison**: How embeddings changed during training

### Real-World Applications

**Semantic Search**
- Find documents with similar meaning (not just keyword matching)
- Example: Query "automobile" finds documents about "car", "vehicle"
- Build intelligent search engines

**Duplicate Detection**
- Identify duplicate or near-duplicate content
- Used in plagiarism detection
- Content deduplication in databases

**Recommendation Systems**
- Find similar products, articles, or content
- "Users who liked X also liked Y" based on embedding similarity
- Content recommendation for streaming services

**Clustering and Grouping**
- Group similar texts together
- Organize documents by topic
- Customer segmentation based on feedback patterns

### Similarity Metrics

**Cosine Similarity** is used to compare embeddings:
- Range: -1.0 to 1.0 (often normalized to 0.0 to 1.0)
- `1.0` = Identical meaning
- `0.8-0.9` = Very similar
- `0.5-0.7` = Moderately similar
- `0.0-0.4` = Different meanings

Formula: `similarity = (A · B) / (||A|| × ||B||)`

**Example:**
```
Text 1: "The cat sat on the mat"
Text 2: "A feline rests on a rug"
→ High cosine similarity (≈0.85) → Very similar meaning
```

---

## 4. 📊 Generic Regression

### What It Does
Predicts continuous numerical values from text. This is the default task and represents general regression problems.

### Output Format
**4-dimensional continuous vector**: `[value1, value2, value3, value4]`

Example outputs:
- `[0.23, -0.45, 0.12, 0.89]` - Raw continuous values
- No direct semantic interpretation (abstract task)

### How to Use
1. Select **"Generic Regression"** from the task dropdown (or leave as default)
2. Enter any text samples
3. The model learns to map text → continuous output vector
4. Useful for understanding the basic mechanics without task-specific interpretation

### Real-World Applications

**Numerical Prediction**
- Predict ratings, scores, or numerical attributes from text
- Estimate quantities, prices, or metrics

**Feature Extraction**
- Generate numerical features for downstream tasks
- Preprocessing step for other machine learning models

**Custom Tasks**
- Use as a template for building your own regression tasks
- Adapt the output dimension for your specific needs

---

## Task Comparison

| Task | Output Dim | Output Type | Interpretation | Use Case |
|------|------------|-------------|----------------|----------|
| **Sentiment** | 2 | Probabilities | Percentages sum to 100% | Review analysis, social media |
| **Classification** | 4 | Probabilities | Highest = predicted category | Email routing, document organization |
| **Similarity** | 8 | Embeddings | Cosine similarity (0-1) | Search, duplicate detection |
| **Regression** | 4 | Continuous | Abstract numerical values | General regression tasks |

---

## Text Encoding

All tasks use **TF-IDF-like feature extraction** to convert text to numerical vectors:

1. **Text Length**: Normalized character count
2. **Word Count**: Number of words
3. **Sentiment Keywords**: Presence of positive/negative/neutral words
4. **Hash-based Features**: Deterministic random features based on text content

**Why this encoding?**
- Simple and fast (no external dependencies)
- Demonstrates the concept without requiring word embeddings
- Deterministic: same text → same encoding
- In production, you'd use pre-trained embeddings (Word2Vec, BERT, etc.)

---

## Interpreting Outputs

### For Classification Tasks (Sentiment, Classification)

**Before Training:**
- Outputs are random/untrained
- Probabilities may not reflect actual text meaning

**After Training:**
- Model learns patterns from training data
- Probabilities align with actual labels
- Higher confidence → clearer predictions

**What to Look For:**
- ✅ Loss decreases → Model is learning
- ✅ Probabilities become more extreme (closer to 0 or 1) → Confidence increases
- ✅ Predictions match expected labels → Model learned correctly

### For Similarity Tasks

**Before Training:**
- Similar texts may have different embeddings
- Cosine similarity is low even for similar content

**After Training:**
- Similar texts produce similar embeddings
- Cosine similarity increases for related content
- Embeddings capture semantic relationships

**What to Look For:**
- ✅ Cosine similarity > 0.7 for similar texts → Good learning
- ✅ Embeddings change meaningfully → Model adapted
- ⚠️ Very high similarity (>0.95) might indicate overfitting

---

## Tips for Best Results

### 1. **Sentiment Analysis**
- ✅ Mix positive and negative examples
- ✅ Use clear, unambiguous sentiment words
- ⚠️ Avoid sarcasm or mixed sentiment (harder to learn)

### 2. **Text Classification**
- ✅ Provide examples from all categories
- ✅ Use diverse sentence structures
- ⚠️ Unbalanced categories may bias predictions

### 3. **Text Similarity**
- ✅ Include pairs of similar texts
- ✅ Include diverse, unrelated texts for contrast
- ✅ Paraphrases work well (same meaning, different words)

### 4. **General**
- ✅ More training epochs → Better learning (but watch for overfitting)
- ✅ Learning rate 0.01 works well for most tasks
- ✅ Enable LoRA for parameter-efficient training
- ✅ Try different LoRA ranks (2, 4, 8) to see trade-offs

---

## Extending Tasks

Want to add your own task? Edit `src/utils/dataset.py`:

1. Add task config to `TaskDataset.TASK_TYPES`
2. Implement `infer_label_from_text()` for your task
3. Add task to dropdown in `src/app.py`
4. Add output interpretation in the results section

Example:
```python
TASK_TYPES = {
    'my_task': {
        'output_dim': 3,
        'labels': ['Label1', 'Label2', 'Label3'],
        'description': 'My custom task description'
    }
}
```

---

## Further Reading

- **Sentiment Analysis**: VADER, TextBlob, BERT-based sentiment models
- **Text Classification**: Naive Bayes, Logistic Regression, Transformers
- **Text Embeddings**: Word2Vec, GloVe, BERT, Sentence Transformers
- **Similarity Metrics**: Cosine similarity, Euclidean distance, Jaccard similarity

For production systems, consider using:
- Pre-trained embeddings (sentence-transformers, spaCy)
- Transformer models (BERT, RoBERTa) for better accuracy
- Proper tokenization and preprocessing pipelines

---

**Note**: This is a **demonstration/toy implementation** for educational purposes. Real-world systems use more sophisticated text encoding (Word2Vec, BERT, etc.) and larger models trained on extensive datasets.

