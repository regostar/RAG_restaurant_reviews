# RAG_restaurant_reviews


## Resources

https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#other-example-llamafiles

https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews

https://huggingface.co/Mozilla/Llama-3.2-3B-Instruct-llamafile/tree/main

https://github.com/alfredodeza/learn-retrieval-augmented-generation/tree/main/examples/1-managing-data

https://www.coursera.org/programs/university-of-north-texas-on-coursera-c8pgo/learn/intro-gen-ai?source=search

# Restaurant Review Analysis & Recommendation System

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-green)](https://qdrant.tech/)
[![Llama-3](https://img.shields.io/badge/LLM-Llama--3.2--3B--Instruct-orange)](https://huggingface.co/Mozilla/Llama-3.2-3B-Instruct-llamafile)

A semantic search system for restaurant reviews using vector embeddings and local LLM-powered recommendations.

## Features

- 🧠 Semantic search using Sentence-BERT embeddings
- 📊 Qdrant vector database for efficient similarity search
- 🦙 Local LLM inference with Llama-3.2-3B-Instruct
- 🍽️ Restaurant review analysis and recommendations
- 🔍 Context-aware query understanding

## Prerequisites

- Python 3.7+
- [Llama-3.2-3B-Instruct.Q6_K.llamafile](https://huggingface.co/Mozilla/Llama-3.2-3B-Instruct-llamafile/tree/main)
- 8GB+ RAM recommended

## Installation

1. **Install Python dependencies**:
```bash
pip install openai sentence-transformers qdrant-client pandas
```

2. **Download llamafile**:

```bash
wget https://huggingface.co/Mozilla/Llama-3.2-3B-Instruct-llamafile/resolve/main/Llama-3.2-3B-Instruct.Q6_K.llamafile
chmod +x Llama-3.2-3B-Instruct.Q6_K.llamafile
```

3. **Start the LLM server**:

```bash
./Llama-3.2-3B-Instruct.Q6_K.llamafile
```

## Code Explanation -

 **Main components :-** 
```bash
    A[Review Data] --> B[Embedding Generation]
    
    B --> C[Qdrant Vector DB]
    
    D[User Query] --> B
    
    C --> E[Semantic Search]
    
    E --> F[Llama-3 LLM]
    
    F --> G[Recommendation]
  ```

## Key Functions

### Data Loading:

```python
# Load CSV data into DataFrame
df = pd.read_csv('restaurant_reviews.csv')
data = df.to_dict('records')
```

### Embedding Generation:

```python
encoder = SentenceTransformer('all-MiniLM-L6-v2')
vector = encoder.encode(review_text).tolist()
```

### Vector Database Setup:

```python
qdrant.recreate_collection(
    collection_name="restaurant_reviews",
    vectors_config=models.VectorParams(
        size=384,  # Dimension of all-MiniLM-L6-v2
        distance=models.Distance.COSINE
    )
)
```

### Semantic Search:

```python
hits = qdrant.search(
    collection_name="restaurant_reviews",
    query_vector=query_embedding,
    limit=3
)
```

### LLM Integration:

```python
response = client.chat.completions.create(
    model="Llama-3.2-3B-Instruct",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": str(context)}
    ],
    temperature=0.7
)
```

### Configuration :

Parameter	Default Value	Description
collection_name	"restaurant_reviews"	Qdrant collection name

embedding_model	"all-MiniLM-L6-v2"	Sentence Transformer model

llm_base_url	"http://localhost:8080/v1"	Llamafile API endpoint

temperature	0.7	LLM creativity control

max_tokens	500	Maximum response length


### Customization

Data: Replace sample data with your CSV file

Prompts: Modify system_prompt in code for different behaviors

Search: Adjust limit and score_threshold in search parameters

