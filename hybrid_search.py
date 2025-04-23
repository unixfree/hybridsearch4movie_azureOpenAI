# Import necessary libraries and modules
import os       
#import argparse
#import time
import openai
import couchbase.search as search
from urllib.parse import urljoin
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AzureOpenAI
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, SearchOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import DocumentNotFoundException
from couchbase.vector_search import VectorQuery, VectorSearch

from datetime import timedelta
import json

#from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.runnables import RunnablePassthrough
#from langchain_core.output_parsers import StrOutputParser

from typing import Any, Dict, List, Tuple
import streamlit as st
#from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.runnables import RunnablePassthrough
#from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Load environment variables
DB_CONN_STR = os.getenv("DB_CONN_STR")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_BUCKET = os.getenv("DB_BUCKET")
DB_SCOPE = os.getenv("DB_SCOPE")
DB_COLLECTION = os.getenv("DB_COLLECTION")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MOVIES_DATASET = "imdb_top_1000.csv"

client = AzureOpenAI(
    api_key = "3ap04zbyLViLshuvfG3dzhErOeAzXQFe5rSgKiVeJ5TCdI9FLFGBJQQJ99BDACHYHv6XJ3w3AAAAACOGA6Py",
    api_version = "2023-12-01-preview",
    azure_endpoint =  "https://kwang-m9sjhrkc-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2025-01-01-preview"
)

def translate_to_english(text):
    response = client.chat.completions.create(
        model="gpt-35-turbo",  # 또는 Azure에 배포한 모델 이름
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates Korean to English."},
            {"role": "user", "content": f"Translate the following text to English. Only return the translated text without any additional comments: {text}"}
        ],
        temperature=0.2,
        max_tokens=100
    )

    return response.choices[0].message.content.strip()

def translate_to_korean(text):
    try:
        response = client.chat.completions.create(
            model="gpt-35-turbo",  # 실제 배포한 Azure 모델 이름
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates English movie descriptions into Korean using safe and neutral language."},
                {"role": "user", "content": f"Translate this movie description into Korean:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=300
        )

        # 응답이 None이 아니고 message.content가 존재할 때만 반환
        if response and response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            return "⚠️ 번역 결과를 생성할 수 없습니다. (응답 없음)"

    except openai.BadRequestError as e:
        if 'content_filter' in str(e):
            return "⚠️ 이 영화 설명은 민감한 콘텐츠로 인해 번역되지 않았습니다."
        else:
            return f"⚠️ 요청 오류 발생: {e}"

    except Exception as e:
        return f"⚠️ 알 수 없는 오류: {e}"

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

@st.cache_resource(show_spinner="Connecting to Couchbase")
def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to couchbase"""

    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    connect_string = connection_string
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster

@st.cache_resource
def create_filter(
    year_range: Tuple[int], rating: float, search_in_title: bool, title: str
) -> Dict[str, Any]:
    """Create a filter for the hybrid search"""
    # Fields in the document used for search
    year_field = "Released_Year"
    rating_field = "IMDB_Rating"
    title_field = "Series_Title"

    filter = {}
    filter_operations = []
    if year_range:
        year_query = {
            "min": year_range[0],
            "max": year_range[1],
            "inclusive_min": True,
            "inclusive_max": True,
            "field": year_field,
        }
        filter_operations.append(year_query)
    if rating:
        filter_operations.append(
            {
                "min": rating,
                "inclusive_min": False,
                "field": rating_field,
            }
        )
    if search_in_title:
        filter_operations.append(
            {
                "match_phrase": title,
                "field": title_field,
            }
        )
    filter["query"] = {"conjuncts": filter_operations}
    return filter

def search_couchbase(
    db_scope: Any,
    index_name: str,
    embedding_key: str,
    search_text: str,
    k: int = 5,
    fields: List[str] = ["*"],
    search_options: Dict[str, Any] = {},
):
    """Hybrid search using Python SDK in couchbase"""
    # Generate vector embeddings to search with
    search_embedding = generate_embedding(search_text)

    # Create the search request
    search_req = search.SearchRequest.create(
        VectorSearch.from_vector_query(
            VectorQuery(
                embedding_key,
                search_embedding,
                k,
            )
        )
    )

    docs_with_score = []

    try:
        # Perform the search
        search_iter = db_scope.search(
            index_name,
            search_req,
            SearchOptions(
                limit=k,
                fields=fields,
                raw=search_options,
            ),
        )

        # Parse the results
        for row in search_iter.rows():
            score = row.score
            docs_with_score.append((row.fields, score))
    except Exception as e:
        raise e

    return docs_with_score


if __name__ == "__main__":
    st.set_page_config(
        page_title="Movie Search",
        page_icon="🎥",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    # Load environment variables
    DB_CONN_STR = os.getenv("DB_CONN_STR")
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_BUCKET = os.getenv("DB_BUCKET")
    DB_SCOPE = os.getenv("DB_SCOPE")
    DB_COLLECTION = os.getenv("DB_COLLECTION")
    INDEX_NAME = os.getenv("INDEX_NAME")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    # Initialize empty filters
    search_filters = {}

    # Connect to Couchbase Vector Store
    cluster = connect_to_couchbase(DB_CONN_STR, DB_USERNAME, DB_PASSWORD)
    bucket = cluster.bucket(DB_BUCKET)
    scope = bucket.scope(DB_SCOPE)

    # UI Elements
    text = st.text_input("Find your movie")
    with st.sidebar:
        st.header("Search Options")
        no_of_results = st.number_input(
            "Number of results", min_value=1, value=5, format="%i"
        )
        # Filters
        st.subheader("Filters")

        year_range = st.slider("Released Year", 1900, 2024, (1900, 2024))
        rating = st.number_input("Minimum IMDB Rating", 0.0, 10.0, 0.0, step=1.0)
        search_in_title = st.checkbox("Search in Title?")
        show_filter = st.checkbox("Show filter")
        hybrid_search_filter = create_filter(
            year_range, rating, search_in_title, text
        )
        if show_filter:
            st.json(hybrid_search_filter)

    if text:
        with st.spinner("Searching..."):

            # 응답에서 번역 결과 추출
            search_text = translate_to_english(text)
            st.write(f"Translated text: {search_text}")

            # Fetch the filters
            search_filters = create_filter(year_range, rating, search_in_title, text)

            # Search using the Couchbase Python SDK
            results = search_couchbase(
                scope,
                INDEX_NAME,
                "Overview_embedding",
                search_text,
                k=no_of_results,
                search_options=search_filters,
            )

            for doc in results:
                movie, score = doc

                # Display the results in a grid
                st.header(movie["Series_Title"])
                col1, col2 = st.columns(2)
                with col1:
                    st.image(movie["Poster_Link"], use_column_width=True)
                with col2:
                    st.write(movie["Overview"])

                    # Translate the movies text to Korean
                    #st.write(translate_to_korean(movie['Overview']).text.strip())
                    st.write(translate_to_korean(movie['Overview']))

                    st.write(f"Score: {score:.{3}f}")
                    st.write("Released Year:", movie["Released_Year"])
                    st.write("IMDB Rating:", movie["IMDB_Rating"])
                    st.write("Runtime:", movie["Runtime"])
                st.divider()
