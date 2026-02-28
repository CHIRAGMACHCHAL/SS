#!/usr/bin/env python3
"""
memory/ingestion.py - Standalone ingestion script for PDFs.
Run with: python -m memory.ingestion --tier public   (or jarvis)
"""

import os
import logging
import hashlib
import argparse
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass
import math
import boto3
from botocore.exceptions import NoCredentialsError
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# =========================
# CONFIGURATION
# =========================
# These should be set via environment variables in production
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

PUBLIC_S3_BUCKET = "mini-agi-public-pdfs"
PUBLIC_S3_PREFIX = "documents/"
JARVIS_S3_BUCKET = "mini-agi-jarvis-pdfs"
JARVIS_S3_PREFIX = "private/"

PUBLIC_COLLECTION = "public_core"
JARVIS_COLLECTION = "jarvis_private"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300

# #=================================================
# #-------FOR LOCALLY-------------------------------
# #================================================
# def calculate_data_fingerprint(data_dir):
#     hasher = hashlib.md5()
#     for file in sorted(os.listdir(data_dir)):
#         if file.lower().endswith(".pdf"):
#             path = os.path.join(data_dir, file)
#             hasher.update(file.encode())
#             hasher.update(str(os.path.getmtime(path)).encode())
#     return hasher.hexdigest()

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]  # ❌ no file logs in Spaces
)
logger = logging.getLogger(__name__)

# =========================
# DOCUMENT DATACLASS
# =========================
@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]

# =========================
# S3 Helpers
# =========================
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

def download_pdfs_from_s3(bucket_name: str, prefix: str = "") -> List[Document]:
    """Download all PDFs from S3 bucket and return list of Document objects (one per page)."""
    s3 = get_s3_client()
    documents = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                if not key.lower().endswith(".pdf"):
                    continue
                logger.info(f"Downloading {key}")
                file_obj = s3.get_object(Bucket=bucket_name, Key=key)
                pdf_reader = PdfReader(file_obj["Body"])
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text() or ""
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": key, "page": page_num}
                    ))
    except NoCredentialsError:
        logger.error("AWS credentials not found.")
        return []
    return documents

# =========================
# Chunking
# =========================
def split_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    separators = ["\n\n", "\n", " "]
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        # try soft split
        for sep in separators:
            idx = chunk.rfind(sep)
            if idx != -1 and idx > chunk_size * 0.5:
                end = start + idx
                chunk = text[start:end]
                break
        chunks.append(chunk.strip())
        if end >= text_length:
            break
        start = max(end - overlap, 0)
    return chunks

def split_documents(documents: List[Document]) -> List[Document]:
    """Split each document page into chunks."""
    all_chunks = []
    for doc in documents:
        text_chunks = split_text_with_overlap(doc.page_content, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(text_chunks):
            all_chunks.append(Document(
                page_content=chunk,
                metadata={**doc.metadata, "chunk": i}
            ))
    logger.info(f"Split into {len(all_chunks)} chunks")
    return all_chunks

# =========================
# Embedding
# =========================
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

# =========================
# Qdrant Helpers
# =========================
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def calculate_s3_fingerprint(bucket_name: str, prefix: str = "") -> str:
    """Calculate MD5 fingerprint of all PDFs in S3 bucket (based on keys and last modified)."""
    s3 = get_s3_client()
    hasher = hashlib.md5()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if "Contents" not in page:
            continue
        for obj in sorted(page["Contents"], key=lambda x: x["Key"]):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                hasher.update(key.encode())
                hasher.update(str(obj["LastModified"]).encode())
                hasher.update(str(obj["Size"]).encode())
    return hasher.hexdigest()

def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False

def upsert_points(client: QdrantClient, collection_name: str, points: List[PointStruct], vector_dim: int = 768):
    MAX_BATCH_BYTES = 4 * 1024 * 1024  # 4MB safe payload size
    BYTES_PER_FLOAT = 4
    approx_vector_bytes = vector_dim * BYTES_PER_FLOAT      # Fix 1: vector_dim use karo
    approx_point_size = approx_vector_bytes + 1500
    BATCH_SIZE = max(50, min(1000, MAX_BATCH_BYTES // approx_point_size))

    total_batches = math.ceil(len(points) / BATCH_SIZE)
    for i in range(0, len(points), BATCH_SIZE):             # Fix 2: BATCH_SIZE use karo
        batch = points[i:i+BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info(f"Upserted batch {i//BATCH_SIZE + 1}/{total_batches}")
#--------------------------------------------------------------------------------------------    

# =========================
# Main Ingestion Function
# =========================
def run_ingestion(tier: str):
    """
    Full ingestion pipeline for given tier.
    - Fingerprint check  → skip if already up-to-date
    - Distributed lock   → safe multi-worker
    - Download → Chunk → Embed → Upsert
    - Smart batch size   → 4MB payload limit aware
    - Optimizer update   → better search speed after ingestion
    """
    # Step 1 — Tier → bucket + collection
    if tier == "public":
        bucket = PUBLIC_S3_BUCKET
        prefix = PUBLIC_S3_PREFIX
        collection = PUBLIC_COLLECTION
    elif tier == "jarvis":
        bucket = JARVIS_S3_BUCKET
        prefix = JARVIS_S3_PREFIX
        collection = JARVIS_COLLECTION
    else:
        raise ValueError(f"Unknown tier: {tier}")

    logger.info(f"Starting ingestion for tier={tier}, bucket={bucket}, collection={collection}")

    qdrant = get_qdrant_client()
    embedder = SentenceTransformerEmbeddings(EMBEDDING_MODEL)
#------------=============------------============------------==================--------------====
    # Compute fingerprint of current S3 contents
    current_fp = calculate_s3_fingerprint(bucket, prefix)
    logger.info(f"Current S3 fingerprint: {current_fp}")

    # Check existing fingerprint in Qdrant collection metadata
    stored_fp = None
    if collection_exists(qdrant, collection):
        coll_info = qdrant.get_collection(collection)
        metadata = getattr(coll_info.config.params, "metadata", {}) or {}
        stored_fp = metadata.get("data_fingerprint")
        logger.info(f"Stored fingerprint: {stored_fp}")

    if stored_fp == current_fp:
        logger.info("Fingerprints match, no ingestion needed.")
        return

    # Distributed lock using Qdrant metadata
    lock_key = "ingestion_lock"       
    # vector_size — ALWAYS compute, dono cases mein
    test_emb = embedder.embed_query("test")
    vector_size = len(test_emb)

    if collection_exists(qdrant, collection):
        coll_info = qdrant.get_collection(collection)
        metadata = getattr(coll_info.config.params, "metadata", {}) or {}
        if metadata.get(lock_key) == "active":
            logger.error("Another worker is already ingesting. Exiting.")
            return
        qdrant.update_collection(
            collection_name=collection,
            metadata={**metadata, lock_key: "active"}
        )
    else:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        metadata = {}
        qdrant.update_collection(
            collection_name=collection,
            metadata={lock_key: "active"}
        )

    try:
        docs = download_pdfs_from_s3(bucket, prefix)
        if not docs:
            logger.warning("No PDFs found.")
            return

        chunks = split_documents(docs)

        texts = [chunk.page_content for chunk in chunks]
        vectors = embedder.embed_documents(texts)
        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk.page_content, **chunk.metadata}
            ))

        upsert_points(qdrant, collection, points, vector_dim=vector_size)  # Fix 3a: vector_dim pass karo

        # Fix 3b: optimizer update — large collections ke liye search speed better hoti hai
        qdrant.update_collection(
            collection_name=collection,
            optimizer_config={}
        )
        # Update metadata with new fingerprint and release lock
        qdrant.update_collection(
            collection_name=collection,
            metadata={"data_fingerprint": current_fp, lock_key: "released"}
        )
        logger.info("Ingestion completed successfully.")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        qdrant.update_collection(
            collection_name=collection,
            metadata={lock_key: "released"}
        )
        raise    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant")
    parser.add_argument("--tier", choices=["public", "jarvis"], required=True,
                        help="Which tier's PDFs to ingest")
    args = parser.parse_args()
    run_ingestion(args.tier)