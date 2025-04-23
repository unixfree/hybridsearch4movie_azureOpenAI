# Import necessary libraries and modules
import os       
import argparse
import time 
import couchbase.search as search
from urllib.parse import urljoin
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AzureOpenAI
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, SearchOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import DocumentNotFoundException
from couchbase.vector_search import VectorQuery, VectorSearch

from datetime import timedelta
from tqdm import tqdm
import uuid
import json
import pandas as pd

import boto3
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from a .env file
load_dotenv()

# Load environment variables
DB_CONN_STR = os.getenv("DB_CONN_STR")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_BUCKET = os.getenv("DB_BUCKET")
DB_SCOPE = os.getenv("DB_SCOPE")
DB_COLLECTION = os.getenv("DB_COLLECTION")
MOVIES_DATASET = "imdb_top_1000.csv"

# Initialize Azure OpenAI client using environment variables
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-10-21",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)   
        
# Function to generate an embedding for a given text using Azure OpenAI
def generate_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model=os.getenv("EMBEDDING_MODEL")
    )
    return response.data[0].embedding

# Initialize Couchbase cluster and collection using environment variables
def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to couchbase"""
    print("Connecting to couchbase...")
    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    connect_string = connection_string
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster

def cleanup_poster_url(poster_url):
    """Convert from https://m.media-amazon.com/images/M/MV5BMDFkYTc0MGEtZmNhMC00ZDIzLWFmNTEtODM1ZmRlYWMwMWFmXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_UX67_CR0,0,67,98_AL_.jpg to https://m.media-amazon.com/images/M/MV5BMDFkYTc0MGEtZmNhMC00ZDIzLWFmNTEtODM1ZmRlYWMwMWFmXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_.jpg"""

    prefix = poster_url.split("_V1_")[0]
    suffix = poster_url.split("_AL_")[1]

    return prefix + suffix

try:
    cluster = connect_to_couchbase(DB_CONN_STR, DB_USERNAME, DB_PASSWORD)
    bucket = cluster.bucket(DB_BUCKET)
    scope = bucket.scope(DB_SCOPE)
    collection = scope.collection(DB_COLLECTION)
    data = pd.read_csv(MOVIES_DATASET)

    # Convert columns to numeric types
    data["Gross"] = data["Gross"].str.replace(",", "").astype(float)

    # Fill empty values
    data["Gross"] = data["Gross"].fillna(0)
    data["Certificate"] = data["Certificate"].fillna("NA")
    data["Meta_score"] = data["Meta_score"].fillna(-1)
    data["Poster_Link"] = data["Poster_Link"].apply(cleanup_poster_url)

    data_in_dict = data.to_dict(orient="records")
    print("Ingesting Data...")
    for row in tqdm(data_in_dict):
        row["Overview_embedding"] = generate_embedding(row["Overview"])
        doc_id = uuid.uuid4().hex
        collection.upsert(doc_id, row)

except Exception as e:
    print("Error while ingesting data", e)
