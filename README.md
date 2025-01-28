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
