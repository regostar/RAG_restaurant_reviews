#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# In[5]:


file = 'Restaurant_reviews.csv'


# In[7]:


# Read data 
df = pd.read_csv(file)
print(df.head())


# In[12]:


df = df.drop("7514", axis=1)
df = df.drop("Pictures", axis=1)
# Check for NA and null values in each column
na_null_check = df.isnull().sum()
print("Null rows :- ", na_null_check )


# In[18]:


df = df[df['Review'].notna()]  # Clean missing reviews
data = df.sample(700).to_dict('records')
data


# In[20]:


# Initialize models
encoder = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(":memory:")

# Create collection for restaurant reviews
qdrant.recreate_collection(
    collection_name="restaurant_reviews",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE
    )
)

# Upload reviews with embeddings
qdrant.upload_points(
    collection_name="restaurant_reviews",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["Review"]).tolist(),
            payload=doc,
        ) for idx, doc in enumerate(data)
    ]
)


# In[22]:


# Example search query
user_prompt = "Find a restaurant for a date night"

# Perform semantic search
hits = qdrant.search(
    collection_name="restaurant_reviews",
    query_vector=encoder.encode(user_prompt).tolist(),
    limit=3
)

# Prepare search results
search_results = [hit.payload for hit in hits]

# Connect to LLM for response generation
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-no-key-required"
)

completion = client.chat.completions.create(
    model="LLaMA_CPP",
    messages=[
        {"role": "system", "content": "You are a restaurant concierge. Help users find dining options based on reviews."},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": str(search_results)}
    ]
)

print(completion.choices[0].message.content)


# In[ ]:




