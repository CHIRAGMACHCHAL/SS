# main.py
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
from pypdf import PdfReader
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
from sentence_transformers import SentenceTransformer
from llm.llm_engine import llm, generate as llm_generate
import hashlib
import boto3
from botocore.exceptions import NoCredentialsError
from memory.conversation_memory import ConversationMemory
from memory.graph_sync import MemoryGraph

# # =========================
# # PUBLIC TIER CONFIGURATION
# # =========================

# PUBLIC_TIER = "free"   # "free" | "pro" | "ultra"

# def apply_public_tier_limits(cognitive_profile: dict) -> dict:
    
#     if PUBLIC_TIER == "free":
#         cognitive_profile["deep_reasoning"] = False
#         cognitive_profile["use_emergent_concepts"] = False
#         cognitive_profile["max_docs"] = 4
#         cognitive_profile["query_complexity"] = "low"
    
#     elif PUBLIC_TIER == "pro":
#         cognitive_profile["deep_reasoning"] = False
#         cognitive_profile["use_emergent_concepts"] = True                                         yyyyyyyyyyyy
#         cognitive_profile["max_docs"] = 8
#         cognitive_profile["query_complexity"] = "normal"
    
#     elif PUBLIC_TIER == "ultra":
#         cognitive_profile["deep_reasoning"] = True
#         cognitive_profile["use_emergent_concepts"] = True
#         cognitive_profile["max_docs"] = 12
#         cognitive_profile["query_complexity"] = "high"
    
#     return cognitive_profile

# =========================
# PRODUCTION QDRANT CONFIG
# =========================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# PUBLIC_COLLECTION = "public_core"
# JARVIS_COLLECTION = "jarvis_private"                                           yyyyyyyyyyyy
# =========================
# QDRANT CLOUD CLIENT (PERMANENT)
# =========================

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
#-----------------------------------------------
# =========================
# AWS S3 CONFIG
# =========================
PUBLIC_S3_BUCKET = "mini-agi-public-pdfs"
PUBLIC_S3_PREFIX = "documents/"

JARVIS_S3_BUCKET = "mini-agi-jarvis-pdfs"
JARVIS_S3_PREFIX = "private/"


# S3_BUCKET_NAME = "mini-agi-public-pdfs"
# S3_PREFIX = "documents/"   # optional folder inside bucket

s3_client = boto3.client("s3")

#------------------------------------------------
  

def calculate_data_fingerprint(data_dir):
    hasher = hashlib.md5()
    for file in sorted(os.listdir(data_dir)):
        if file.lower().endswith(".pdf"):
            path = os.path.join(data_dir, file)
            hasher.update(file.encode())
            hasher.update(str(os.path.getmtime(path)).encode())
    return hasher.hexdigest()


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]


#----------------------------------------------------------------
# Configure logging                                                                                       #parmanent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]  # ❌ no file logs in Spaces
)

# Constants
# DATA_DIR = r"D:\OLLAMA\data"


# # ===== SYSTEM MODE =====
# SYSTEM_MODE = "public"   # "public" or "jarvis"                                                      yyyyyyyyyyyy

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
#-----------------------------------------------
# VECTOR_STORE_NAME = "simple-rag"
# def get_collection_name(system_mode: str):
#     if system_mode == "jarvis":                                                                   yyyyyyyyyyyy
#         return JARVIS_COLLECTION
#     return PUBLIC_COLLECTION
#-------------------------------------------
#-----------------------------------------------
def download_pdfs_from_s3(bucket_name, prefix=""):
    """
    Downloads PDFs from S3 into temporary memory
    Returns list of Document objects
    """
    documents = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        
            if "Contents" not in page:
                continue
        
            for obj in page["Contents"]:
        
                key = obj["Key"]
        
                if not key.lower().endswith(".pdf"):
                    continue
        
                file_obj = s3_client.get_object(
                    Bucket=bucket_name,
                    Key=key
                )
        
        # response = s3_client.list_objects_v2(
        #     Bucket=bucket_name,
        #     Prefix=prefix
        # )

        # if "Contents" not in response:
        #     logging.info("No files found in S3 bucket.")
        #     return []

        # for obj in response["Contents"]:
        #     key = obj["Key"]

        #     if key.lower().endswith(".pdf"):
        #         file_obj = s3_client.get_object(
        #             Bucket=bucket_name,
        #             Key=key
        #         )

                pdf_reader = PdfReader(file_obj["Body"])

                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text() or ""

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": key,
                                "page": page_num
                            }
                        )
                    )

                logging.info(f"Loaded from S3: {key}")

    except NoCredentialsError:
        logging.error("AWS credentials not found.")
        return []

    return documents

#----------------------------------------------------

def ingest_pdfs_from_folder(data_dir):
    """ALL PDFs from a folder (LangChain-free, identical output)"""
    all_docs = []

    for file in os.listdir(data_dir):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file)

            reader = PdfReader(pdf_path)

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""

                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file,
                        "page": page_num
                    }
                )

                all_docs.append(doc)

            logging.info(f"Loaded PDF: {file}")

    return all_docs

def split_text_with_overlap(text: str, chunk_size: int, overlap: int):
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

def split_documents(documents):
    """LangChain-free document splitter (identical behavior)"""
    chunk_size = 1200
    overlap = 300

    all_chunks = []

    for doc in documents:
        text_chunks = split_text_with_overlap(
            doc.page_content,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for i, chunk in enumerate(text_chunks):
            new_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk": i
                }
            )
            all_chunks.append(new_doc)

    logging.info("Documents split into chunks.")
    return all_chunks
#                      step 4
class SentenceTransformerEmbeddings:
    """
    Drop-in replacement for OllamaEmbeddings
    Interface SAME rakha gaya hai
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()
    

class QdrantVectorStore:
    def __init__(self, qdrant_client, collection_name, embedding_model):
        self.qdrant = qdrant_client
        self.collection = collection_name
        self.embedder = SentenceTransformerEmbeddings(embedding_model)


    def similarity_search(self, query: str, k: int = 4):
        # query_vector = self.embedder.embed_query(query)
        query_vector = self.embedder.embed_query(query)

        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=k
        )

        docs = []
        for r in results:
            docs.append(
                Document(
                    page_content=r.payload["text"],
                    metadata=r.payload
                )
            )

        return docs

def vector_db_has_data(collection_name):
    try:
        info = qdrant.get_collection(collection_name)
        return info.points_count > 0
    except Exception:
        return False

def calculate_s3_fingerprint(bucket_name, prefix=""):
    hasher = hashlib.md5()

    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if "Contents" not in page:
            continue

        for obj in sorted(page["Contents"], key=lambda x: x["Key"]):
            key = obj["Key"]
            hasher.update(key.encode())
            hasher.update(str(obj["LastModified"]).encode())
            hasher.update(str(obj["Size"]).encode())

    return hasher.hexdigest()


def create_or_load_vector_db(data_dir=None, system_mode="public"):
    # ollama.pull(EMBEDDING_MODEL)

    collection_name = get_collection_name(system_mode)
    # fingerprint_file = "./qdrant_data/data.fingerprint"
    # current_fp = calculate_data_fingerprint(data_dir)
    # current_fp = calculate_s3_fingerprint(S3_BUCKET_NAME, S3_PREFIX)
    if system_mode == "jarvis":
        bucket = JARVIS_S3_BUCKET
        prefix = JARVIS_S3_PREFIX
    else:
        bucket = PUBLIC_S3_BUCKET
        prefix = PUBLIC_S3_PREFIX
    
    current_fp = calculate_s3_fingerprint(bucket, prefix)


    # Collection exists + fingerprint match
    # if vector_db_has_data(collection_name) and os.path.exists(fingerprint_file):
    #     with open(fingerprint_file, "r") as f:
    #         saved_fp = f.read().strip()

    #     if saved_fp == current_fp:
    #         logging.info("Qdrant vector DB up-to-date. Loading only.")
    #         return QdrantVectorStore(
    #             qdrant_client=qdrant,
    #             collection_name=collection_name,
    #             embedding_model=EMBEDDING_MODEL
    #         )
    stored_fp = None

    if vector_db_has_data(collection_name):
        info = qdrant.get_collection(collection_name)
    
        metadata = getattr(info.config.params, "metadata", None)
    
        if metadata and isinstance(metadata, dict):
            stored_fp = metadata.get("data_fingerprint")

    # stored_fp = None

    # if vector_db_has_data(collection_name):
    #     info = qdrant.get_collection(collection_name)
    #     stored_fp = info.config.params.metadata.get("data_fingerprint")
    
    
    if stored_fp == current_fp:
        logging.info("Qdrant vector DB up-to-date. Loading only.")
        return QdrantVectorStore(
            qdrant_client=qdrant,
            collection_name=collection_name,
            embedding_model=EMBEDDING_MODEL
        )

    logging.info("Qdrant DB missing or outdated. Rebuilding...")
    #----------------------------------------------------------------
    # ===== Distributed Lock Using Qdrant Metadata =====
    lock_key = "ingestion_lock"
    
    existing_collections = [
        c.name for c in qdrant.get_collections().collections
    ]
    
    if collection_name in existing_collections:
        collection_info = qdrant.get_collection(collection_name)
        metadata = getattr(collection_info.config.params, "metadata", {}) or {}
    else:
        metadata = {}

    # collection_info = qdrant.get_collection(collection_name)
    # metadata = getattr(collection_info.config.params, "metadata", {}) or {}
    
    if metadata.get(lock_key) == "active":
        logging.info("Another worker is rebuilding. Waiting...")
        raise RuntimeError("Ingestion already in progress by another instance.")
    
    # Activate lock
    qdrant.update_collection(
        collection_name=collection_name,
        metadata={**metadata, lock_key: "active"}
    )
    
    try:
        # ---- ingestion code yahan se ----
        documents = download_pdfs_from_s3(...)
        ...
        qdrant.upsert(...)
        
        qdrant.update_collection(
            collection_name=collection_name,
            metadata={
                "data_fingerprint": current_fp,
                "ingestion_lock": "released"
            }
        )
    
    finally:
        # Safety release
        qdrant.update_collection(
            collection_name=collection_name,
            metadata={"ingestion_lock": "released"}
        )

    #---------------------------------------------------------------

    # documents = ingest_pdfs_from_folder(data_dir)
    documents = download_pdfs_from_s3(
        bucket_name= bucket,
        prefix=prefix
    )

    #----------------------------------------------------
    if not documents:
        raise ValueError("No PDFs found to ingest")

    chunks = split_documents(documents)
    embedder = SentenceTransformerEmbeddings(EMBEDDING_MODEL)
    vector_size = len(embedder.embed_query("test"))
    
    existing_collections = [
        c.name for c in qdrant.get_collections().collections
    ]
    
    if collection_name not in existing_collections:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        logging.info(f"Collection '{collection_name}' created.")
    else:
        logging.info(f"Collection '{collection_name}' already exists.")

    # 🔹 create collection if not exists
    # if not qdrant.collection_exists(collection_name):
    #     qdrant.create_collection(
    #         collection_name=collection_name,
    #         vectors_config=VectorParams(
    #             size=768,  #
    #             distance=Distance.COSINE
    #         )
    #     )

    #---------------------------------------------------------------------
    # embedder = SentenceTransformerEmbeddings(EMBEDDING_MODEL)
    # vector_size = len(embedder.embed_query("test"))
    
    # existing_collections = [
    #     c.name for c in qdrant.get_collections().collections
    # ]
    
    # if collection_name not in existing_collections:
    #     qdrant.create_collection(
    #         collection_name=collection_name,
    #         vectors_config=VectorParams(
    #             size=vector_size,
    #             distance=Distance.COSINE
    #         )
    #     )
        


    #-------------------------------------------------------------------------
    points = []

    texts = [doc.page_content for doc in chunks]
    vectors = embedder.embed_documents(texts)
    
    for doc, vector in zip(chunks, vectors):
    
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": doc.page_content,
                    **doc.metadata
                }
            )
        )


    # points = []
    # for doc in chunks:
    #     vector = embedder.embed_documents([doc.page_content])[0]

    #     # vector = embedder.embed_query(doc.page_content)

    #     points.append(
    #         PointStruct(
    #             id=str(uuid.uuid4()),
    #             vector=vector,
    #             payload={
    #                 "text": doc.page_content,
    #                 **doc.metadata
    #             }
    #         )
    #     )

    # qdrant.upsert(
    #     collection_name=collection_name,
    #     points=points
    # )
    #--------------------------------------
    # BATCH_SIZE = 100
    import math

    MAX_BATCH_BYTES = 4 * 1024 * 1024  # 4MB safe payload size
    VECTOR_DIM = vector_size
    BYTES_PER_FLOAT = 4
    
    approx_vector_bytes = VECTOR_DIM * BYTES_PER_FLOAT
    approx_payload_bytes = 1500  # average text metadata estimate
    approx_point_size = approx_vector_bytes + approx_payload_bytes
    
    BATCH_SIZE = max(50, min(1000, MAX_BATCH_BYTES // approx_point_size))


    #------------------------------------

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]
    
        qdrant.upsert(
            collection_name=collection_name,
            points=batch
        )

    #----------------------------------------------------
    qdrant.update_collection(
        collection_name=collection_name,
        optimizer_config={}
    )

    qdrant.update_collection(
        collection_name=collection_name,
        metadata={
            "data_fingerprint": current_fp,
            "ingestion_lock": "released"
        }
    )

    logging.info("Qdrant vector DB created/updated.")
    
    return QdrantVectorStore(
        qdrant_client=qdrant,
        collection_name=collection_name,
        embedding_model=EMBEDDING_MODEL
    )
#=========================================================================================================================================
#..............Phase 2.5 : CONTEXTUAL ENRICHER ..........
# class MemoryGraphAdapter:
#     """
#     Wraps vector DB into a reasoning-aware memory graph.
#     """

#     def __init__(self, vector_db):
#         self.vector_db = vector_db

#     def estimate_relevance(self, query: str) -> float:
#         try:
#             docs = self.vector_db.similarity_search(query, k=3)
#             if not docs:
#                 return 0.0
#             return min(1.0, len(docs) / 3)
#         except Exception:
#             return 0.3

#     def retrieve(self, query: str, k: int = 5):
#         return self.vector_db.similarity_search(query, k=k)
class MemoryGraphAdapter:
    """
    Hybrid Memory Graph:
    - Vector retrieval
    - Emergent concept linking
    - Lightweight relational activation
    """

    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.concept_graph = {}

    def _extract_concepts(self, text: str):
        words = text.lower().split()
        return [w for w in words if len(w) > 6]

    def _update_graph(self, docs):
        for doc in docs:
            concepts = self._extract_concepts(doc.page_content)

            for c in concepts:
                if c not in self.concept_graph:
                    self.concept_graph[c] = set()

                for related in concepts:
                    if related != c:
                        self.concept_graph[c].add(related)

    def retrieve(self, query: str, k: int = 6):
        docs = self.vector_db.similarity_search(query, k=k)

        if not docs:
            return []

        # Build graph from retrieved docs
        self._update_graph(docs)

        # Activate related concepts
        activated = set()
        query_words = self._extract_concepts(query)

        for qw in query_words:
            if qw in self.concept_graph:
                activated.update(self.concept_graph[qw])

        # Expand retrieval using activated concepts
        expanded_query = query + " " + " ".join(list(activated)[:5])

        return self.vector_db.similarity_search(expanded_query, k=k)

    def estimate_relevance(self, query: str) -> float:
        docs = self.retrieve(query, k=3)
        return min(1.0, len(docs) / 3)

    

# =========================
# VECTOR DB + MEMORY GRAPH INIT
# =========================
vector_db = create_or_load_vector_db(None, system_mode=SYSTEM_MODE) #public ya jarvis
memory_graph = MemoryGraphAdapter(vector_db)



# # =========================
# # LAYER 4: MEMORY GRAPH
# # =========================

# class MemoryGraph:
#     def __init__(self):
#         self.graph = {}

#     def add_concept(self, concept, related_concepts):
#         self.graph[concept] = related_concepts

#     def activate(self, question):
#         activated = []

#         for concept, relations in self.graph.items():
#             if concept.lower() in question.lower():
#                 activated.append(concept)
#                 activated.extend(relations)

#         return list(set(activated))

#====== Very Important ========= 

# =========================
# PHASE 2: IMPLICIT / EMERGENT MEMORY
# =========================

def implicit_memory_retrieval(vector_db, question, k=12):
    """
    Phase-2 implicit memory:
    - No hardcoded concepts
    - Embedding similarity drives memory
    - Concepts emerge from retrieved chunks
    """

    # Step 1: Raw semantic retrieval
    docs = vector_db.similarity_search(question, k=k)

    # Step 2: Extract emergent keywords (lightweight signal)
    concept_counter = {}

    for doc in docs:
        words = doc.page_content.lower().split()
        for w in words:
            if len(w) > 5:   # noise filter
                concept_counter[w] = concept_counter.get(w, 0) + 1

    # Step 3: Top emergent "concept hints"
    emergent_concepts = sorted(
        concept_counter,
        key=concept_counter.get,
        reverse=True
    )[:8]

    return docs, emergent_concepts


# =========================
# PHASE 2.1: COGNITIVE ROUTER
# =========================


class CognitiveRouter:
    def route(self, question: str) -> str:
        q = question.lower()

        if self.is_memory_query(q):
            return "memory"

        if self.is_fact_query(q):
            return "retrieval"

        if self.is_reasoning_query(q):
            return "reasoning"

        return "direct"
    
    def route_with_context(
            self,
            *,
            question: str,
            intent,
            domains,
            required_depth
        ) -> str:
            """
            FULL cognitive routing (heavy, non-lite)
                    """
    
            # 1️⃣ Intent based routing
            if intent in {"reasoning", "planning", "comparison"}:
                primary_route = "reasoning_engine"
    
            elif intent in {"factual", "definition", "lookup"}:
                primary_route = "knowledge_retrieval"
    
            elif intent in {"creative", "story", "idea"}:
                primary_route = "creative_engine"
    
            else:
                primary_route = "hybrid_engine"
    
            # 2️⃣ Domain override
            if domains:
                if "science" in domains or "tech" in domains:
                    primary_route = "knowledge_retrieval"
    
                elif "philosophy" in domains:
                    primary_route = "reasoning_engine"
    
            # 3️⃣ Depth modulation
            if required_depth == "deep":
                if primary_route == "knowledge_retrieval":
                    primary_route = "hybrid_engine"
    
            # 4️⃣ Safety fallback (old router)
            if not primary_route:
                primary_route = self.route(question)
    
            return primary_route

    def is_memory_query(self, q):
        return any(x in q for x in [
            "yaad", "pehle", "tumne kaha", "memory", "earlier"
        ])

    def is_fact_query(self, q):
        return any(x in q for x in [
            "what is", "who", "when", "define", "list"
        ])

    def is_reasoning_query(self, q):
        return any(x in q for x in [
            "why", "how", "explain", "kaise", "kyu"
        ])


def memory_lookup(vector_db, question, k=6):
    docs = vector_db.similarity_search(question, k=k)
    if not docs:
        return None
    return "\n".join(doc.page_content for doc in docs[:3])


# =========================
# PHASE 2.5: RESPONSE STRATEGY ENGINE
# =========================

class ResponseStrategyEngine:
    def decide(self, route, intent, mode, cognitive_profile):
        """
        Decides HOW the answer should be framed
        Returns a response strategy dict
        """

        strategy = {
            "style": "neutral",
            "structure": "plain",
            "verbosity": "medium",
            "system_prompt": None
        }

        # ===== PUBLIC MODE =====
        if mode == "public":
            if route == "retrieval":
                strategy.update({
                    "style": "informative",
                    "structure": "concise",
                    "verbosity": "low"
                })

            elif route == "reasoning":
                strategy.update({
                    "style": "clear",
                    "structure": "step-lite",
                    "verbosity": "medium"
                })

        # ===== JARVIS MODE =====
        if mode == "jarvis":
            if intent == "research":
                strategy.update({
                    "style": "analytical",
                    "structure": "sectioned",
                    "verbosity": "high"
                })

            elif intent == "execution":
                strategy.update({
                    "style": "instructional",
                    "structure": "steps",
                    "verbosity": "high"
                })

            elif intent == "conversation":
                strategy.update({
                    "style": "casual",
                    "structure": "free",
                    "verbosity": "low"
                })

        return strategy


# =========================
# PHASE 2.6: RESPONSE ASSEMBLY ENGINE
# =========================

class ResponseAssemblyEngine:
    def assemble(self, raw_answer: str, strategy: dict):
        """
        Shapes the final answer based on response strategy
        WITHOUT changing factual content
        """

        answer = raw_answer.strip()

        # ===== Verbosity Control =====
        if strategy["verbosity"] == "low":
            answer = " ".join(answer.split()[:80])

        elif strategy["verbosity"] == "medium":
            answer = " ".join(answer.split()[:150])

        # ===== Structural Control =====
        if strategy["structure"] == "steps":
            answer = "Steps:\n" + answer

        elif strategy["structure"] == "sectioned":
            answer = "Analysis:\n" + answer

        # ===== Style Injection (light, non-invasive) =====
        if strategy["style"] == "analytical":
            answer = "Here is a detailed analysis:\n\n" + answer

        elif strategy["style"] == "instructional":
            answer = "Follow these instructions carefully:\n\n" + answer

        return answer

# =========================
# PHASE 3: WORLD MODEL ENGINE
# =========================

class WorldModelEngine:
    def analyze(self, question: str, intent: str, cognitive_profile: dict):
        q = question.lower()

        world = {
            "domain": "general",
            "time_sensitivity": "timeless",
            "human_factor": False,
            "ethical_weight": "low",
            "power_dynamics": False,
            "detected_intent": intent  # Layer 1 ka output yahan integrate ho gaya
        }

        # ---- Domain detection ----
        if any(x in q for x in ["ai", "model", "algorithm", "code", "llm"]):
            world["domain"] = "technology"

        elif any(x in q for x in ["king", "prince", "power", "state", "rule"]):
            world["domain"] = "political"

        elif any(x in q for x in ["emotion", "feel", "love", "anger"]):
            world["domain"] = "psychological"
            world["human_factor"] = True

        # ---- Time sensitivity ----
        if any(x in q for x in ["current", "today", "now", "recent"]):
            world["time_sensitivity"] = "current"

        # ---- Ethics / power ----
        if any(x in q for x in ["control", "manipulate", "influence"]):
            world["power_dynamics"] = True
            world["ethical_weight"] = "medium"

        # Blueprint logic: Agar profile 'deep_reasoning' hai toh complexity badhao
        if cognitive_profile.get("deep_reasoning"):
            world["ethical_weight"] = "high"
            world["reasoning_depth"] = "max"    

        return world


# =========================
# PHASE 3.1: DYNAMIC WORLD ASSUMPTIONS
# =========================

class WorldAssumptionEngine:
    def enrich(self, world_state: dict, domains: list):
        assumptions = {
            "bias_risk": "low",
            "response_tone": "neutral",
            "ambiguity_allowed": False,
            "domain_context": domains # Blueprint trace
        }

        if world_state["domain"] == "political":
            assumptions.update({
                "bias_risk": "high",
                "ambiguity_allowed": True,
                "response_tone": "balanced"
            })

        if world_state["domain"] == "psychological":
            assumptions.update({
                "response_tone": "empathetic",
                "ambiguity_allowed": True
            })

        if world_state["domain"] == "technology":
            assumptions.update({
                "response_tone": "precise"
            })

        world_state["assumptions"] = assumptions
        return world_state

# =========================
# PHASE 3.2: WORLD-MEMORY BINDING
# =========================

class WorldMemoryBinder:
    def bind(self, world_state: dict, emergent_concepts: list):
        filtered = emergent_concepts.copy()

        # Political → remove emotional noise
        if world_state["domain"] == "political":
            filtered = [
                c for c in filtered
                if c not in ["feel", "emotion", "anger"]
            ]

        # Psychological → allow emotions
        if world_state.get("human_factor"):
            filtered = emergent_concepts

        # Tech → precision bias
        if world_state["domain"] == "technology":
            filtered = filtered[:5]

        return filtered

# =========================
# PHASE 3.3: CONFLICT DETECTION
# =========================

class ConflictDetector:
    def analyze(self, docs):
        signals = {
            "conflict": False,
            "uncertainty": False
        }

        if len(docs) < 2:
            return signals

        contents = [doc.page_content.lower() for doc in docs]

        keywords = ["however", "but", "on the other hand", "although"]

        for text in contents:
            if any(k in text for k in keywords):
                signals["conflict"] = True
                signals["uncertainty"] = True
                break

        return signals

# =========================
# PHASE 3.4: WORLD-AWARE QUERY MUTATION
# =========================

class WorldQueryMutator:
    def mutate(self, question: str, world_state: dict, expanded_queries: list = None):
        q = expanded_queries[0] if expanded_queries else question.strip()

        if world_state["domain"] == "political":
            q += " considering power structures and accountability"

        if world_state["domain"] == "psychological":
            q += " focusing on emotional and behavioral aspects"

        if world_state["domain"] == "technology":
            q += " with technical accuracy and implementation details"

        return q


# =========================
# PHASE 4.0: SELF STATE ENGINE
# =========================

class SelfStateEngine:
    def build(self, *, mode, intent, route, world_state):
        state = {
            "mode": mode,
            "intent": intent,
            "route": route,
            "world_domain": world_state["domain"],
            "confidence_level": "unknown",
            "capability_mode": "base",
            "risk_tolerance": "medium"
        }

        if mode == "jarvis":
            state["capability_mode"] = "extended"
            state["risk_tolerance"] = "low"

        if intent == "research":
            state["risk_tolerance"] = "very_low"

        return state


# =========================
# PHASE 4.1: CAPABILITY AWARENESS
# =========================

class CapabilityAwarenessEngine:
    def evaluate(self, self_state):
        capabilities = {
            "can_reason_deep": False,
            "can_give_opinion": False,
            "needs_caution": False
        }

        if self_state["capability_mode"] == "extended":
            capabilities["can_reason_deep"] = True

        if self_state["world_domain"] in ["political", "psychological"]:
            capabilities["needs_caution"] = True

        if self_state["intent"] == "conversation":
            capabilities["can_give_opinion"] = True

        self_state["capabilities"] = capabilities
        return self_state

# =========================
# PHASE 4.2: SELF CONFIDENCE ESTIMATION
# =========================

class SelfConfidenceEngine:
    def estimate(self, self_state, world_state):
        confidence = "medium"

        if world_state["domain"] == "technology":
            confidence = "high"

        if world_state["domain"] == "political":
            confidence = "low"

        if self_state["intent"] == "research":
            confidence = "medium"

        self_state["confidence_level"] = confidence
        return self_state

# =========================
# PHASE 2.2: COGNITIVE LOAD CONTROLLER
# =========================

class CognitiveLoadController:
    def decide(self, route: str, mode: str,world_state: dict, intent: dict | None = None, required_depth: str = "normal"):
        """
        Returns cognitive profile based on route + system mode
        """

        # Default (public safe)
        profile = {
            "use_chain": True,
            "deep_reasoning": False,
            "use_emergent_concepts": False,
            "max_docs": 5
        }

        if mode == "public":
            if route == "reasoning":
                profile["deep_reasoning"] = False
            return profile

        # ===== JARVIS MODE =====
        if mode == "jarvis":
            if route == "reasoning":
                profile.update({
                    "deep_reasoning": True,
                    "use_emergent_concepts": True,
                    "max_docs": 10
                })

            elif route == "retrieval":
                profile.update({
                    "use_emergent_concepts": True,
                    "max_docs": 8
                })

            elif route == "memory":
                profile.update({
                    "use_chain": False
                })

        # ---- World-aware adjustments ----
        if world_state["domain"] == "political":
            profile["deep_reasoning"] = True
            profile["use_emergent_concepts"] = True

        if world_state["human_factor"]:
            profile["use_chain"] = False  # avoid cold logic

        if world_state["ethical_weight"] == "medium":
            profile["deep_reasoning"] = True

        # ---- Intent-aware tuning (Blueprint-aligned) ----
        if intent:
            if intent.get("urgency") == "high":
                profile["deep_reasoning"] = True

        if required_depth == "deep":
            profile["deep_reasoning"] = True
            profile["max_docs"] += 2

        return profile
# def expand_query(llm, question):
#     """
#     ChatGPT-style query expansion:
#     - preserves intent
#     - extracts key concepts
#     - adds semantic variants
#     """

#     prompt = f"""
# You are an internal query-expansion module.

# Task:
# Rewrite the user question into ONE expanded search query
# that preserves the original intent but includes:
# - key concepts
# - important synonyms
# - implicit angles needed for deep retrieval

# Rules:
# - Do NOT change the meaning
# - Do NOT introduce unrelated domains
# - Do NOT answer the question
# - Output ONE expanded query sentence only

# User question:
# {question}
# """

#     response = llm.invoke(prompt)

#     return (
#         response.content
#         .strip()
#         .strip('"')
#         .replace("\n", " ")
#     ) 
# def retrieve_docs_expanded(vector_db, expanded_query, k=8):
#     return vector_db.similarity_search(expanded_query, k=k)


# =========================
# PHASE 4.3: SELF REFINEMENT DECISION
# =========================

class SelfRefinementEngine:
    def decide(self, self_state, cognitive_profile):
        if self_state["confidence_level"] == "low":
            cognitive_profile["deep_reasoning"] = True
            cognitive_profile["use_emergent_concepts"] = True

        if self_state["confidence_level"] == "high":
            cognitive_profile["use_chain"] = False

        return cognitive_profile

# =========================
# PHASE 4.4: HEAVY META-COGNITION ENGINE
# =========================

class HeavyMetaCognitionEngine:
    def evaluate(self, *, question, self_state, world_state, cognitive_profile):
        meta_flags = {
            "slow_thinking": False,
            "multi_perspective": False,
            "assumption_check": False,
            "ethical_reflection": False
        }

        # High-risk or abstract domains
        if world_state["domain"] in ["political", "psychological"]:
            meta_flags["slow_thinking"] = True
            meta_flags["multi_perspective"] = True
            meta_flags["assumption_check"] = True

        # Power / influence / control
        if world_state.get("power_dynamics"):
            meta_flags["ethical_reflection"] = True
            meta_flags["slow_thinking"] = True

        # Low confidence → force deeper cognition
        if self_state["confidence_level"] == "low":
            meta_flags["slow_thinking"] = True
            meta_flags["assumption_check"] = True

        # ---- Apply meta decisions to cognition ----
        if meta_flags["slow_thinking"]:
            cognitive_profile["deep_reasoning"] = True
            cognitive_profile["use_chain"] = True

        if meta_flags["multi_perspective"]:
            cognitive_profile["use_emergent_concepts"] = True

        return cognitive_profile, meta_flags

# =========================
# Phase 4.5 : Meta-Control Engine
# =========================
class MetaControlEngine:
    def decide(self, question, cognitive_profile, meta_flags, self_state):
        override = False
        updated_profile = cognitive_profile

        if meta_flags.get("overthinking"):
            updated_profile["deep_reasoning"] = False
            override = True

        if meta_flags.get("uncertain"):
            updated_profile["max_docs"] = min(
                updated_profile.get("max_docs", 6) + 2,
                12
            )
            override = True

        return type(
            "MetaDecision",
            (),
            {
                "override": override,
                "updated_profile": updated_profile
            }
        )()

# =========================
# Phase 4.6 : Meta-Retry Engine
# =========================
class MetaRetryEngine:
    def check(self, cognitive_profile, self_state):
        should_retry = False

        if cognitive_profile.get("confidence", 1.0) < 0.4:
            should_retry = True

        return type(
            "RetrySignal",
            (),
            {
                "should_retry": should_retry
            }
        )()

    def adjust(self, cognitive_profile):
        cognitive_profile["deep_reasoning"] = True
        cognitive_profile["max_docs"] = min(
            cognitive_profile.get("max_docs", 6) + 2,
            14
        )
        return cognitive_profile


# Phase 5.1 — Goal Formation

class GoalFormationEngine:
    def infer(self, question, intent, world_state):
        return {
            "goal": question,
            "intent": intent,
            "world_context": world_state
        }


# Phase 5.2 — Agency Safety

class AgencySafetyEngine:
    def evaluate(self, goal, mode):
        if mode != "jarvis":
            return {"allow": False, "reason": "Agency disabled"}
        return {"allow": True}
# Phase 5.3 — Plan Synthesis

class PlanSynthesisEngine:
    def build(self, goal, world_state):
        return [
            {"step": 1, "name": "understand_goal"},
            {"step": 2, "name": "reason"},
            {"step": 3, "name": "respond"}
        ]

# Phase 5.4 — Action Selection

class ActionSelectionEngine:
    def select(self, plan, cognitive_profile):
        actions = []
        for p in plan:
            actions.append({
                "action": p["name"],
                "priority": "high" if cognitive_profile.get("deep_reasoning") else "normal"
            })
        return actions

# Phase 5.5 — Tool Invocation

class ToolInvocationEngine:
    def invoke(self, actions):
        results = []
        for action in actions:
            results.append({
                "action": action["action"],
                "status": "done"
            })
        return results
# Phase 5.6 — Execution Monitoring

class ExecutionMonitorEngine:
    def evaluate(self, results):
        success = all(r["status"] == "done" for r in results)
        return {
            "success": success,
            "results": results
        }


# =========================
# PHASE 6A: ALIGNMENT FINETUNING
# =========================

class AlignmentFineTuner:
    def evaluate(self, *, question, answer, meta, agency_result, mode):
        if mode != "jarvis":
            return None

        alignment = {
            "clarity_ok": True,
            "safety_ok": True,
            "overreach": False,
            "confidence_level": meta.get("confidence")
        }

        if agency_result and agency_result.get("blocked"):
            alignment["safety_ok"] = True

        if meta.get("confidence") == "low":
            alignment["clarity_ok"] = False

        if "must" in answer.lower() or "always" in answer.lower():
            alignment["overreach"] = True

        return alignment
# =========================
# PHASE 6B: KNOWLEDGE FINETUNING
# =========================

class KnowledgeFineTuner:
    def update(self, *, question, world_state, cognitive_profile, alignment_report):
        memory_patch = {
            "question_pattern": question[:120],
            "domain": world_state.get("domain"),
            "used_deep_reasoning": cognitive_profile.get("deep_reasoning"),
            "alignment": alignment_report
        }

        # NOTE:
        # Abhi sirf simulation hai
        # Future me:
        # → vector DB write
        # → model adapters
        # → LoRA / memory graph

        logging.info("[PHASE 6] Knowledge patch stored")
        return memory_patch

class SimplePrompt:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

class SimpleParser:
    def parse(self, output) -> str:
        # Ollama / transformers output → string
        return str(output)

class SimpleChain:
    def __init__(self, llm, prompt: SimplePrompt, parser: SimpleParser):
        self.llm = llm
        self.prompt = prompt
        self.parser = parser

    def invoke(self, inputs: dict) -> str:
        formatted_prompt = self.prompt.format(**inputs)

        # LLM call (same as LangChain behavior)
        raw_output = self.llm.invoke(formatted_prompt)

        return self.parser.parse(raw_output)

def create_chain(llm):
    """
    LangChain-free replacement for:
        prompt | llm | StrOutputParser
    Returns a chain object ready to run with context/question at runtime
    """
    template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""

    # Prompt object
    prompt = SimplePrompt(template)
    parser = SimpleParser()

    # Chain object
    chain = SimpleChain(llm=llm, prompt=prompt, parser=parser)

    logging.info("Chain created successfully.")
    return chain


# =========================
# PHASE 2.3: META-COGNITION / RETRY ENGINE / SELF-EVALUATION
# =========================

class MetaCognitionEngine:
    def evaluate(self, answer: str, mode: str):
        """
        Returns confidence score and decision
        """

        # Simple heuristic (lightweight, no heavy logic)
        length = len(answer.split())

        confidence = "low"
        retry = False

        if length > 120:
            confidence = "high"
        elif length > 60:
            confidence = "medium"

        # Jarvis is self-critical
        if mode == "jarvis" and confidence != "high":
            retry = True

        return {
            "confidence": confidence,
            "retry": retry
        }
    

# =========================
# PHASE 2.4: INTENT STATE ENGINE
# =========================

class IntentStateEngine:

    def detect(self, question: str) -> str:
        """
        Fallback single-intent detection (legacy / safety)
        """
        q = question.lower()

        if any(k in q for k in ["research", "study", "analyze", "compare", "why"]):
            return "research"
        if any(k in q for k in ["do", "build", "create", "execute", "run"]):
            return "execution"
        if any(k in q for k in ["how", "what", "explain", "define"]):
            return "information"

        return "general"

    # ==================================================
    # 🔥 NEW METHOD — LAYER 1 AWARE INTENT DETECTION
    # ==================================================
    def detect_from_layer1(self, layer1_bundle: dict) -> str:
        """
        Uses Layer-1 intent decomposition output to infer
        the dominant intent state.

        Input example:
        layer1_bundle = {
            "intent_type": "mixed",
            "sub_goals": [...],
            "domains": [...],
            "required_depth": "deep"
        }
        """

        if not layer1_bundle:
            return "general"

        intent_type = layer1_bundle.get("intent_type", "general")
        sub_goals = layer1_bundle.get("sub_goals", [])
        required_depth = layer1_bundle.get("required_depth", "normal")

        # 1️⃣ Direct mapping (strongest signal)
        if intent_type in ["research", "analysis"]:
            return "research"

        if intent_type in ["execution", "action"]:
            return "execution"

        # 2️⃣ Multi-intent → research bias
        if intent_type == "mixed":
            if required_depth in ["deep", "very_deep"]:
                return "research"
            return "information"

        # 3️⃣ Sub-goal heuristic
        for goal in sub_goals:
            g = goal.lower()
            if any(k in g for k in ["compare", "evaluate", "analyze"]):
                return "research"
            if any(k in g for k in ["build", "implement", "execute"]):
                return "execution"

        # 4️⃣ Safe default
        return "information"


# =========================
# PHASE 2.7: SAFETY / CONSTRAINT LAYER
# =========================

class SafetyConstraintEngine:
    def evaluate(self, *, mode, intent, route, question):
        """
        Rule-based safety & capability gating
        No RLHF, no emotion, no softness
        """

        decision = {
            "allow": True,
            "reason": None
        }

        # 🔒 Public restrictions
        if mode == "public":
            if intent in ["execution", "research"]:
                decision["allow"] = False
                decision["reason"] = "Restricted capability in public mode"

        # 🔥 Explicit dangerous signals (expand later)
        dangerous_keywords = [
            "weapon", "bomb", "explosive", "build missile",
            "chemical synthesis", "harm", "kill"
        ]

        if any(k in question.lower() for k in dangerous_keywords):
            decision["allow"] = False
            decision["reason"] = "Potentially dangerous request"

        return decision


# =========================
# PHASE 2.8: TRACE LOGGER
# =========================

class TraceLogger:
    def log(self, trace: dict):
        """
        Internal cognitive trace
        Should NEVER be exposed to public users
        """

        logging.info("===== TRACE LOG =====")
        for k, v in trace.items():
            logging.info(f"{k}: {v}")
        logging.info("=====================")

# =========================
# STEP-1 TRAINING CONFIG                                                 
# =========================       
ENABLE_TRAINING = False   # ❌ abhi OFF Future me ise "True" kr deng jb training krenge
TRAINING_MODE = "offline"  # offline | lora | full_finetune

TRAINING_DATA_DIR = "./training/datasets"
TRAINING_LOG_DIR = "./training/logs"

#=========================================
# STEP-2 UNIVERSAL TRAINING INTERFACE (CORE)
#=========================================
#📌 Is file ko future me kabhi rewrite nahi karna
# 📌 Sirf andar logic upgrade hoga
class TrainingEngine:
    """
    UNIVERSAL TRAINING INTERFACE
    ----------------------------
    - Dataset agnostic
    - Model agnostic
    - Future-proof (LoRA / full finetune)
    """

    def __init__(self, llm, mode="offline"):
        self.llm = llm
        self.mode = mode

    def load_dataset(self, dataset_path):
        """
        Dataset format (JSONL):
        {
          "instruction": "...",
          "input": "...",
          "output": "..."
        }
        """
        import json

        samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    def train_step(self, sample):
        """
        SINGLE training interaction
        (later replaced by gradient update)
        """
        prompt = f"""
Instruction:
{sample['instruction']}

Input:
{sample['input']}

Expected Understanding:
{sample['output']}
"""
        return self.llm.invoke(prompt)

    def run_training(self, dataset_path):
        samples = self.load_dataset(dataset_path)

        for idx, sample in enumerate(samples):
            self.train_step(sample)

        return {
            "trained_on_samples": len(samples),
            "mode": self.mode,
            "status": "completed"
        }
      
#==============================================================
#🧱 STEP 3 — PHASE-6 TRAINING CONTROLLER (HOOK)
#==============================================================
class Phase6TrainingController:
    """
    Phase-6 Training Orchestrator
    -----------------------------
    - Controls WHEN training happens
    - Keeps architecture clean
    """

    def __init__(self, training_engine):
        self.engine = training_engine

    def maybe_train(self, dataset_path):
        if not ENABLE_TRAINING:
            return {
                "skipped": True,
                "reason": "Training disabled by config"
            }

        return self.engine.run_training(dataset_path)

# ---- Phase 6C : Training Hook (DISABLED BY DEFAULT) ----
        

training_engine = TrainingEngine(
    llm=llm,
    mode=TRAINING_MODE
)

training_controller = Phase6TrainingController(training_engine)

if ENABLE_TRAINING:
    training_result = training_controller.maybe_train(
        dataset_path="./training/datasets/sample.jsonl"
    )
    logging.info(f"[PHASE 6 TRAINING] → {training_result}")

# =========================================
# LAYER 3 — KNOWLEDGE SOURCE ROUTING (HEART)
# =========================================


# -------- Phase 3.0 : Question Classification --------
class KnowledgeSourceClassifier:
    def classify(self, question: str) -> str:
        q = question.lower()

        if any(x in q for x in ["how", "steps", "process", "method"]):
            return "procedural"
        if any(x in q for x in ["who", "when", "where", "date"]):
            return "factual"
        if any(x in q for x in ["why", "explain", "theory", "philosophy"]):
            return "conceptual"

        return "general"


# -------- Phase 3.1 : Source Priority Resolution --------
class SourcePriorityResolver:
    def resolve(self, category: str) -> dict:
        if category == "factual":
            return {"memory": True, "retrieval": True, "reasoning": False}

        if category == "procedural":
            return {"memory": False, "retrieval": True, "reasoning": True}

        if category == "conceptual":
            return {"memory": True, "retrieval": False, "reasoning": True}

        return {"memory": True, "retrieval": True, "reasoning": True}


# -------- Phase 3.2 : Confidence Gating --------
class ConfidenceGate:
    def apply(self, routing: dict, memory_score: float) -> dict:
        if memory_score < 0.4:
            routing["memory"] = False
            routing["reasoning"] = True
        return routing
# -------- Phase 3.4 : Ambiguity Detection --------
class AmbiguityDetector:
    def detect(self, question: str) -> bool:
        vague_terms = ["something", "things", "stuff", "about", "etc"]
        q = question.lower()
        return any(term in q for term in vague_terms)
# -------- Phase 3.5 : Source Conflict Resolution --------
class SourceConflictResolver:
    def resolve(self, routing: dict) -> dict:
        if routing["memory"] and routing["retrieval"]:
            routing["reasoning"] = True
        return routing
# -------- Phase 3.6 : Hallucination Guard --------
class HallucinationGuard:
    def apply(self, routing: dict, confidence: float) -> dict:
        if routing["category"] == "factual" and confidence < 0.3:
            routing["memory"] = False
            routing["reasoning"] = False
            routing["retrieval"] = True
        return routing


# -------- Phase 3.3 : Final Knowledge Router --------
class KnowledgeRouter:
    def route(self, question: str, memory_score: float = 0.5, cognitive_profile: dict = None) -> dict:
        classifier = KnowledgeSourceClassifier()
        resolver = SourcePriorityResolver()
        gate = ConfidenceGate()

        category = classifier.classify(question)
        routing = resolver.resolve(category)
        routing = gate.apply(routing, memory_score)

        return {
            "use_memory": routing["memory"],
            "use_retrieval": routing["retrieval"],
            "use_reasoning": routing["reasoning"],
            "category": category,
            "confidence": memory_score
        }


# ===============================
# Layer 1 : Intent Decomposition Engine
# ===============================

class IntentDecompositionEngine:
    def __init__(self):
        pass

    def process(self, user_query: str, state: dict) -> dict:
        """
        Input:
            user_query : raw user question
            state      : shared thinking state
        Output:
            updated state with decomposed intents
        """
        #========================================
        #=====Phase 1.0 : Raw Query Capture =====
        #========================================
        layer1_state = {}
        
        layer1_state["raw_query"] = user_query

        #======================================
        # =====Phase 1.1 : Linguistic Normalization =====
        # ======================================
        class QueryNormalizer:
            def normalize(self, question: str) -> str:
                return question.strip().lower()
        
        normalizer = QueryNormalizer()
        normalized_query = normalizer.normalize(user_query)
        
        layer1_state["normalized_query"] = normalized_query
        #========================================
        # =====Phase 1.2 : Intent Type Detection (thinking type) =====
        # ========================================
        class IntentTypeDetector:
            def detect(self, q: str) -> str:
                if any(x in q for x in ["how", "steps", "process"]):
                    return "procedural"
                if any(x in q for x in ["why", "meaning", "philosophy"]):
                    return "philosophical"
                if any(x in q for x in ["is it true", "real", "myth"]):
                    return "analytical"
                if any(x in q for x in ["right", "wrong", "ethics", "dharma"]):
                    return "ethical"
                return "general"
        
        intent_detector = IntentTypeDetector()
        intent_type = intent_detector.detect(normalized_query)
        
        layer1_state["intent_type"] = intent_type   
     
        
        #====================================================
        # =====Phase 1.3 : Goal Decomposition =====
        #===================================================
        class GoalDecomposer:
            def decompose(self, q: str, intent_type: str):
                goals = []
        
                goals.append("understand_user_claim")
        
                if intent_type in ["analytical", "philosophical"]:
                    goals.extend([
                        "identify_sources",
                        "check_evidence",
                        "compare_with_modern_knowledge",
                        "separate belief_from_fact"
                    ])
        
                if intent_type == "ethical":
                    goals.append("ethical_evaluation")
        
                return goals
        
        decomposer = GoalDecomposer()
        sub_goals = decomposer.decompose(normalized_query, intent_type)
        
        layer1_state["sub_goals"] = sub_goals

        
        #===================================================
        # =====Phase 1.4 : Query Expansion  =====
        #=================================================
        class QueryExpander:
            def expand(self, q: str, sub_goals: list):
                expansions = [q]
        
                if "identify_sources" in sub_goals:
                    expansions.append(q + " ancient texts sources")
        
                if "compare_with_modern_knowledge" in sub_goals:
                    expansions.append(q + " modern science comparison")
        
                if "separate_belief_from_fact" in sub_goals:
                    expansions.append(q + " evidence based analysis")
        
                return expansions
        
        expander = QueryExpander()
        expanded_queries = expander.expand(normalized_query, sub_goals)
        
        layer1_state["expanded_queries"] = expanded_queries

        
        #=================================================
        # =====Phase 1.5 : Reasoning Depth Estimation =====
        #==================================================
        class ReasoningDepthEstimator:
            def estimate(self, sub_goals: list):
                if len(sub_goals) > 3:
                    return "deep"
                return "medium"
        
        depth_engine = ReasoningDepthEstimator()
        required_depth = depth_engine.estimate(sub_goals)
        
        layer1_state["required_depth"] = required_depth

        #=======================================================
        # =====Phase 1.6 : Knowledge domain mapping (vedic + modern) =====
        #=======================================================
        class KnowledgeDomainMapper:
            def map(self, q: str):
                domains = []
        
                if any(x in q for x in ["veda", "purana", "dharma", "viman"]):
                    domains.append("scriptural")
        
                if any(x in q for x in ["science", "technology", "physics"]):
                    domains.append("scientific")
        
                domains.append("general_knowledge")
        
                return list(set(domains))
        
        domain_mapper = KnowledgeDomainMapper()
        domains = domain_mapper.map(normalized_query)
        
        layer1_state["domains"] = domains

        
        #========================================================
        #======== Phase 1.7 : Intent Bundle ========
        #=======================================================
        intent_bundle = {
            "raw_query": layer1_state["raw_query"],
            "normalized_query": normalized_query,
            "intent_type": intent_type,
            "sub_goals": sub_goals,
            "expanded_queries": expanded_queries,
            "required_depth": required_depth,
            "domains": domains
        }

        state["layer1_intent_bundle"] = intent_bundle
        return state
    

class MemoryAwarePruner:
    """
    Prunes expanded queries using memory relevance.
    """

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def prune(self, queries: list[str], memory_graph) -> list[str]:
        pruned = []
        for q in queries:
            score = memory_graph.estimate_relevance(q)
            if score >= self.threshold:
                pruned.append(q)
        return pruned

class MemoryAwareQueryPruner:
    def prune(self, queries, memory_graph, intent_state):
        """
        Remove redundant or low-value queries
        based on:
        - memory similarity
        - intent criticality
        """
        pruned = []
        seen = set()

        for q in queries:
            score = memory_graph.estimate_relevance(q)

            key = q.lower().strip()

            if key not in seen:
                seen.add(key)
                pruned.append(q)

        return pruned

#================================================
#==========LAYER 2 : ADAPTIVE QUERY EXPANSION(DYNAMICS)==========
#================================================
# .......Phase 2.1 : INTENT-wise QUERY BRANCHING.........
class IntentQueryBrancher:
    def branch(self, intent_state, sub_goals):
        branches = []

        if intent_state == "research":
            for goal in sub_goals:
                branches.append({
                    "type": "exploratory",
                    "goal": goal
                })

        elif intent_state == "execution":
            branches.append({
                "type": "procedural",
                "goal": "steps / how-to"
            })

        elif intent_state == "explanation":
            branches.append({
                "type": "conceptual",
                "goal": "why / theory"
            })

        else:
            branches.append({
                "type": "general",
                "goal": "overview"
            })

        return branches
#.............. Phase 2.2 : QUERY GRANULARITY DECISION ..........
class QueryGranularityDecider:
    def decide(self, intent_state, required_depth):
        if required_depth == "deep" or intent_state == "research":
            return "fine"
        elif required_depth == "broad":
            return "coarse"
        return "normal"

#.................. Phase 2.3 : DYNAMIC QUERY SHAPE GENERATOR ..........
class QueryShapeGenerator:
    def generate(self, base_question, branch, granularity):
        if branch["type"] == "exploratory":
            return f"{base_question} focusing on {branch['goal']} in detail"

        if branch["type"] == "procedural":
            return f"Step-by-step guide for {base_question}"

        if branch["type"] == "conceptual":
            return f"Theoretical explanation of {base_question}"

        return base_question
#...............Phase 2.4 : ABSTRACTION LEVEL MODULATOR ..........
class AbstractionModulator:
    def adjust(self, query, granularity):
        if granularity == "fine":
            return query + " with technical precision and edge cases"

        if granularity == "coarse":
            return "High-level overview of " + query

        return query


#..........Phase 2.6 : QUERY PRIORITY & BUDGET ALLOCATION ..........
class QueryBudgetAllocator:
    def allocate(self, queries, cognitive_profile):
        budget = cognitive_profile.get("max_docs", 6)
        prioritized = sorted(
            queries,
            key=lambda q: len(q),
            reverse=True
        )
        return prioritized[:budget]
#............Phase 2.7 : FINAL QUERY BUNDLE OUTPUT..........
class AdaptiveQueryExpansionEngine:
    def run(
        self,
        question,
        layer1_bundle,
        intent_state,
        cognitive_profile,
        memory_graph
    ):
        brancher = IntentQueryBrancher()
        granularity_decider = QueryGranularityDecider()
        shape_gen = QueryShapeGenerator()
        abstraction = AbstractionModulator()
        pruner = MemoryAwarePruner()
        allocator = QueryBudgetAllocator()

        branches = brancher.branch(
            intent_state,
            layer1_bundle.get("sub_goals", [])
        )

        granularity = granularity_decider.decide(
            intent_state,
            layer1_bundle.get("required_depth", "normal")
        )

        queries = []
        for b in branches:
            q = shape_gen.generate(question, b, granularity)
            q = abstraction.adjust(q, granularity)
            queries.append(q)

        queries = pruner.prune(queries, memory_graph)
        queries = allocator.allocate(queries, cognitive_profile)
        
        queries = MemoryAwareQueryPruner().prune(
                    queries,
                    memory_graph,
                    intent_state
                )



        return queries


#===========================================================
#==========LAYER 5 : REASONING & SYNTHESIS =================
#===========================================================

#............Phase 5.1 - Evidence Aggregation Engine........
class EvidenceAggregator:
    def collect(
        self,
        retrieved_docs=None,
        memory=None,
        reasoning_output=None,
        world_state=None,
        agency_result=None
    ):
        return {
            "retrieval": retrieved_docs or [],
            "memory": memory,
            "reasoning": reasoning_output,
            "world_state": world_state,
            "agency": agency_result
        }

#............Phase 5.2 - Contradiction Detection Engine...............
class ContradictionDetector:
    def detect(self, evidence_bundle):
        conflicts = []

        retrieval = evidence_bundle.get("retrieval", [])
        memory = evidence_bundle.get("memory")

        if memory and retrieval:
            for doc in retrieval:
                if memory in doc.page_content:
                    continue
                conflicts.append({
                    "type": "memory_vs_retrieval",
                    "detail": "Memory and retrieved docs mismatch"
                })

        return {
            "has_conflict": len(conflicts) > 0,
            "conflicts": conflicts
        }

#................Phase 5.3 - Source Trust Scoring............
class SourceTrustScorer:
    def score(self, evidence_bundle, cognitive_profile, route=None ):
        score = {
            "memory": 0.7,
            "retrieval": 0.7,
            "reasoning": 0.8
        }

        if cognitive_profile.get("deep_reasoning"):
            score["reasoning"] += 0.1

        if cognitive_profile.get("confidence", 0.6) < 0.6:
            score["retrieval"] += 0.1

        if route == "memory":
            score["memory"] += 0.1
        elif route == "retrieval":
            score["retrieval"] += 0.1
    
        
        return score

#................Phase 5.4 - Reconcillation Engine...................
class ReconciliationEngine:
    def reconcile(self, evidence_bundle, conflicts, trust_scores):
        if not conflicts["has_conflict"]:
            return evidence_bundle["reasoning"]

        # Conflict hai → weighted decision
        if trust_scores["reasoning"] >= trust_scores["retrieval"]:
            return evidence_bundle["reasoning"]

        return "Re-evaluated answer based on stronger evidence"

#.............Phase 5.5 - Answer Structuring Engine..................
class AnswerStructurer:
    def structure(self, resolved_answer, intent_state):
        if intent_state == "research":
            return f"""
### Summary
{resolved_answer}

### Key Points
- Evidence-backed reasoning
- Conflicts resolved
- Sources reconciled
"""
        return resolved_answer

#................Phase 5.6 - Self-Judgement Engine.........
class SelfJudgeEngine:
    def evaluate(self, answer, cognitive_profile):
        score = 0.7
        if cognitive_profile.get("deep_reasoning"):
            score += 0.1

        return {
            "score": score,
            "acceptable": score >= 0.75
        }

#................Phase 5.7 - Final Synthesis Engine.............
class FinalSynthesisEngine:
    def synthesize(
        self,
        evidence_bundle,
        conflicts,
        trust_scores,
        intent_state,
        cognitive_profile
    ):
        reconciler = ReconciliationEngine()
        base_answer = reconciler.reconcile(
            evidence_bundle, conflicts, trust_scores
        )

        structurer = AnswerStructurer()
        structured = structurer.structure(base_answer, intent_state)

        judge = SelfJudgeEngine()
        judgement = judge.evaluate(structured, cognitive_profile)

        return structured, judgement
    
#----------------------------------------------------
# =========================
# SECTION 7 — OUTPUT BOUNDARY & RESPONSE GUARD
# =========================

class OutputBoundaryGuard:
    """
    FINAL RESPONSE BOUNDARY
    -----------------------
    - Runs AFTER Layer-5 synthesis
    - Does NOT modify reasoning logic
    - Only enforces output safety + policy
    """

    def enforce(self, *, answer: str, mode: str, intent: str, cognitive_profile: dict = None) -> str:
        final_answer = answer.strip()

        # ---- Public hard limits ----
        if mode == "public":
            # ❌ No commands / authority tone
            forbidden_phrases = [
                "you must",
                "you should always",
                "it is mandatory",
                "guaranteed",
                "100%"
            ]
            for p in forbidden_phrases:
                if p in final_answer.lower():
                    final_answer = final_answer.replace(p, "")
                    final_answer += f"\n\n[Boundary Notice: authority tone adjusted for public mode]"


        
            # ❌ No excessive certainty
            if "always" in final_answer.lower():
                final_answer += "\n\n(Note: This may vary depending on context.)"
        



        # ---- Jarvis mode transparency ----
        if mode == "jarvis" and intent == "research":
            final_answer += "\n\n— Generated with extended reasoning enabled."

        return final_answer

# =========================
# SECTION 8 — DEPLOYMENT GOVERNANCE
# =========================

class DeploymentGovernor:
    """
    Deployment & Tier Governance
    ----------------------------
    - Controls exposure
    - NOT part of cognition
    """

    def apply(self, *, mode: str, intent: str, response: str) -> str:
        # ---- Public tier constraints ----
        if mode == "public":
            if intent in ["research", "execution"]:
                response = (
                    "[Public Notice] Response adjusted for public tier visibility.\n\n"
                    + response
                )

        response = (
            f"[Public Tier: {PUBLIC_TIER.upper()}]\n\n"
            + response
        )


        # ---- Jarvis audit tag ----
        if mode == "jarvis":
            response += "\n\n[Governance: Jarvis-tier execution]"

        return response


   
#========================================================================================================================
#============================================================================================================================
    
async def main():
    


    training_result = training_controller.maybe_train(
        dataset_path="./training/datasets/sample.jsonl"
    )

    logging.info(f"[PHASE 6 TRAINING] → {training_result}")

    # vector_db = create_or_load_vector_db(DATA_DIR)

    # Initialize the language model
          

    chain = create_chain(llm)

    question = "How to report The Prince ?"

    memory_layer = ConversationMemory()  
    conversation_id = str(uuid.uuid4())  

    # Layer 4 Full Memory Graph (Blueprint ka REAL BRAIN - Persistent + Tier-aware)
    memory_graph_full = MemoryGraph()
    await memory_graph_full.init_connections()

    # Layer 8 + Layer 4 combined initialization complete

     # Layer 8 connections initialize (async)
    await memory_layer.init_connections()

    # ================================
    # LAYER 1 : INTENT DECOMPOSITION
    # ================================
    
    intent_engine = IntentDecompositionEngine()
    
    state = {}
    state = intent_engine.process(
        user_query=question,
        state=state
    )
    layer1_bundle = state.get("layer1_intent_bundle", {})
    # state["layer1_intent_bundle"] now contains:
    # - intent_type
    # - sub_goals
    # - expanded_queries
    # - domains
    # - required_depth
    #----------------------------------------------------------------
    # Layer-1 Hard Guarantees (Production Safety)
    layer1_bundle.setdefault("normalized_query", question)           #parmanent
    layer1_bundle.setdefault("thinking_type", "mixed")
    layer1_bundle.setdefault("reasoning_plan", [])

    #----------------------------------------------------------------

    # ===== Phase 2.4 : Intent State =====
    intent_engine = IntentStateEngine()
    intent_state = intent_engine.detect_from_layer1(layer1_bundle)
    
    logging.info(f"[Intent State] → {intent_state}")

    # ================================
    # LAYER 2A — COGNITIVE ROUTER
    # ================================
    
    router = CognitiveRouter()
    
    cognitive_route = router.route_with_context(
        question=question,
        intent=intent_state,
        domains=layer1_bundle.get("domains", []),
        required_depth=layer1_bundle.get("required_depth", "normal")
    )
    
    logging.info(f"[Cognitive Route] → {cognitive_route}")
    #-------------------------------------------------------------
    world_state = {
    "domain": None,
    "human_factor": False,
    "ethical_weight": "low"
    }

    #--------------------------------------------------------------
    # ================================
    # LAYER 2B — COGNITIVE LOAD CONTROLLER
    # ================================
    
    load_controller = CognitiveLoadController()
    
    cognitive_profile = load_controller.decide(
        route=cognitive_route,
        mode=SYSTEM_MODE,
        world_state=world_state,
        intent=intent_state,
        required_depth=layer1_bundle.get("required_depth", "normal")
        
    )
    # ===== 🔒 LAYER-2 HARD GUARANTEES (ADD THIS) =====
    cognitive_profile.setdefault("deep_reasoning", False)
    cognitive_profile.setdefault("use_emergent_concepts", False)
    cognitive_profile.setdefault("max_docs", 6)
    cognitive_profile.setdefault("query_complexity", "normal")

    # ===== Layer-2 Confidence Bootstrap =====
    cognitive_profile.setdefault(
        "confidence",
        0.6 if cognitive_profile.get("deep_reasoning") else 0.75
    )
    logging.info(f"[Cognitive Profile] → {cognitive_profile}")
    # Ab system ko pata hai :
          #1. kitna deep sochna hai | 2.fast/ slow/ research mode | 3.token + reasoning budget | layer 3 isi output pr chalegi

    # ===== PUBLIC TIER ENFORCEMENT =====
    if SYSTEM_MODE == "public":
        cognitive_profile = apply_public_tier_limits(cognitive_profile)
      
    
    
       # ================================
    # LAYER 2 — ADAPTIVE QUERY EXPANSION
    # ================================
    
    adaptive_query_engine = AdaptiveQueryExpansionEngine()
    
    adaptive_queries = adaptive_query_engine.run(
        question=question,
        layer1_bundle=layer1_bundle,
        intent_state=intent_state,
        cognitive_profile=cognitive_profile,
        memory_graph=memory_graph   # Layer 4 hook
    )
    #---------------------------------------------------
    # 🔒 Layer-2 Production Lock
    if not isinstance(adaptive_queries, (dict, list)):                     #parmanent
        raise RuntimeError("Layer-2 output corrupted. Blueprint violation.")

    #-----------------------------------------------
    
    logging.info(f"[Adaptive Queries] → {adaptive_queries}")
    
    #--------------------------------------------------------------------------
       

    # ===== Layer-2 → Router Bridge (Blueprint Compliant Fix) =====
    try:
        # Agar adaptive_queries Dictionary hai (Best for AGI Blueprint)
        adaptive_query_text = (
            " ".join(
                q for group in adaptive_queries.values()
                for q in group
            )
            if adaptive_queries else ""
        )

    except AttributeError:
        # Fallback: Agar engine ne sirf ek simple List bheji hai
        adaptive_query_text = " ".join(adaptive_queries)
    
    logging.info(f"Bridge Active: Combined Query for Layer-3 -> {adaptive_query_text}")

    # # ===== Layer-2 → Router Bridge =====
    # adaptive_query_text = " ".join(
    #     q for group in adaptive_queries.values()
    #     for q in group
    # )
    # # ===== Layer-2 Adaptive Feedback =====
    # if sum(len(v) for v in adaptive_queries.values()) > 6:
    #     cognitive_profile["deep_reasoning"] = True
    #     cognitive_profile["query_complexity"] = "high"
    
    query_count = sum(len(v) for v in adaptive_queries.values()) if isinstance(adaptive_queries, dict) else len(adaptive_queries)
    
    if query_count > 6 and not cognitive_profile.get("deep_reasoning"):
        cognitive_profile["deep_reasoning"] = True
        cognitive_profile["query_complexity"] = "high"


    # if query_count > 6:
    #     cognitive_profile["deep_reasoning"] = True
    #     cognitive_profile["query_complexity"] = "high"

    # 🔁 Layer-2 → Cognitive Sync
    if cognitive_profile.get("deep_reasoning"):
        cognitive_profile["max_docs"] = max(
            cognitive_profile.get("max_docs", 6), 8
        )

    # ================================
    # LAYER 3A — WORLD MODEL
    # ================================
    
    world_engine = WorldModelEngine()
    
    world_state = world_engine.analyze(
        question=question,
        intent=intent_state,
        cognitive_profile=cognitive_profile
    )
    # ===== Layer-2 → Layer-3 Cognitive Contract =====
    world_state["cognitive_confidence"] = cognitive_profile.get("confidence", 0.6)
    world_state["reasoning_depth"] = cognitive_profile.get("deep_reasoning", False)
    world_state["allowed_uncertainty"] = (
        "high" if cognitive_profile.get("deep_reasoning") else "medium"
    )

    logging.info(f"[World Model] → {world_state}")
    
    
    # ================================
    # LAYER 3B — WORLD ASSUMPTIONS
    # ================================
    
    assumption_engine = WorldAssumptionEngine()
    
    world_state = assumption_engine.enrich(
        world_state,
        domains=layer1_bundle.get("domains", [])
    )
    
    logging.info(f"[World Assumptions] → {world_state}")
    
    
    # ================================
    # LAYER 3C — QUERY MUTATION
    # ================================
    
    query_mutator = WorldQueryMutator()
    
    mutated_question = query_mutator.mutate(
        question,
        world_state,
        expanded_queries=layer1_bundle.get("expanded_queries", [])
    )
    
    logging.info(f"[Mutated Question] → {mutated_question}")
     
    # ab system janta hai :
                       #1. hidden assumptions | 2.missing context | 3. user ne jo bola,wo bhi  
    # ──────────────── Layer 8: Fetch previous conversation history ────────────────
    history = await memory_layer.get_conversation_history(
        conversation_id=conversation_id,
        tier=SYSTEM_MODE
    )

    # History ko current context mein mix kar do (very useful for continuity)
    if history.strip():
        mutated_question = f"Previous conversation context:\n{history}\n\nCurrent question: {mutated_question}"

    # Optional: History ko mutated_question mein mix kar sakte ho
    # mutated_question = f"{history}\n\nCurrent query: {mutated_question}"

 
    # ===== Phase 3.6 : FINAL KNOWLEDGE ROUTING (ONLY CALL) =====
    router3 = KnowledgeRouter()
    knowledge_route = router3.route(
        question=(
            mutated_question + " " +
            " ".join(layer1_bundle.get("expanded_queries", [])) + " " + adaptive_query_text  # 👈 VERY IMPORTANT
        ),
        memory_score=cognitive_profile.get("confidence", 0.6),
        cognitive_profile=cognitive_profile
    )
    
    final_route = (
        "memory" if knowledge_route["use_memory"]
        else "retrieval" if knowledge_route["use_retrieval"]
        else "reasoning"
    )
   
    
    
    
    # routing_decision decides:
    # - use_memory
    # - use_retrieval
    # - use_reasoning
    
    
    
    

    

    logging.info(f"[Router Decision] → {final_route}")

    # ===== Phase 2.5 : Response Strategy =====
    strategy_engine = ResponseStrategyEngine()
    response_strategy = strategy_engine.decide(
        route=final_route,
        intent={
            "state": intent_state,
            "required_depth": layer1_bundle.get("required_depth", "normal")
        },
        mode=SYSTEM_MODE,
        cognitive_profile=cognitive_profile
    )

    logging.info(f"[Response Strategy] → {response_strategy}")


    if SYSTEM_MODE == "jarvis":
        cognitive_profile["deep_reasoning"] = True
        cognitive_profile["use_emergent_concepts"] = True
        cognitive_profile["max_docs"] = 15


    
    # if SYSTEM_MODE == "jarvis" and intent_state == "research":
    
    #     cognitive_profile.update({
    #     "deep_reasoning": True,
    #     "max_docs": 12
    #     })

    
     # ===== Phase 2.7 : Safety / Constraint =====
    safety_engine = SafetyConstraintEngine()
    safety = safety_engine.evaluate(
        mode=SYSTEM_MODE,
        intent=intent_state,
        route=final_route,
        question=question
    )

    logging.info(f"[Safety Check] → {safety}")

    if not safety["allow"]:
        print(f"Request blocked: {safety['reason']}")
        return

    
                
        
        
    
    # ======= Routed Cognition ========    
    if final_route == "memory":
        memory = memory_lookup(vector_db, question)
        if memory:
            final_prompt = f"""
Use the following memory context to answer carefully:

{memory}

Question:
{question}
"""
            res = llm_generate(final_prompt)
        else:
            res = "No relevant memory found."

    elif final_route == "retrieval":

        docs = []
        memory = None
    

            # Phase 3 complete ho chuka
        
        # ===== Phase 4.0 =====
        self_state_engine = SelfStateEngine()
        self_state = self_state_engine.build(
            mode=SYSTEM_MODE,
            intent=intent_state,
            route=final_route,
            world_state=world_state
        )
        
        # ===== Phase 4.1 =====
        cap_engine = CapabilityAwarenessEngine()
        self_state = cap_engine.evaluate(self_state)
        
        # ===== Phase 4.2 =====
        confidence_engine = SelfConfidenceEngine()
        self_state = confidence_engine.estimate(self_state, world_state)
        
        # ===== Phase 4.3 =====
        refinement_engine = SelfRefinementEngine()
        cognitive_profile = refinement_engine.decide(
            self_state,
            cognitive_profile
        )
        
        # ===== Phase 4.4 : Heavy Meta-Cognition =====
        heavy_meta_engine = HeavyMetaCognitionEngine()
        cognitive_profile, meta_flags = heavy_meta_engine.evaluate(
            question=question,
            self_state=self_state,
            world_state=world_state,
            cognitive_profile=cognitive_profile
        )
        
        logging.info(f"[Heavy Meta] → {meta_flags}")

        

        # ============================
        # Phase 4.5 — Meta-Control Loop
        # ============================
        
        meta_control = MetaControlEngine()
        
        decision = meta_control.decide(
            question=question,
            cognitive_profile=cognitive_profile,
            meta_flags=meta_flags,
            self_state=self_state
        )
        
        # Apply meta-level override if required
        if decision.override:
            cognitive_profile = decision.updated_profile   
                
                    
        
        
        # ==================================
        # Phase 4.6 — Meta-Failure / Retry
        # ==================================
        
        meta_retry = MetaRetryEngine()
        
        retry_signal = meta_retry.check(
            cognitive_profile=cognitive_profile,
            self_state=self_state
        )
        
        # Adjust cognition if retry is required
        if retry_signal.should_retry:
            cognitive_profile = meta_retry.adjust(cognitive_profile)

        

        # 🔁 Phase 4 effect propagation
        max_docs = cognitive_profile["max_docs"]
        use_emergent = cognitive_profile.setdefault("use_emergent_concepts", False)
        deep_reasoning = cognitive_profile.setdefault("deep_reasoning", False)

        

        # ====== Phase 2 : Implicit Memory =====
        docs, emergent_concepts = implicit_memory_retrieval(
            vector_db,
            mutated_question,
            k=10
        )
        
        # ===== Phase 3.3 : Conflict Detection =====
        conflict_detector = ConflictDetector()
        conflict_signals = conflict_detector.analyze(docs)

        if conflict_signals["conflict"] and not cognitive_profile.get("deep_reasoning"):
            cognitive_profile["deep_reasoning"] = True


        # if conflict_signals["conflict"]:
        #     cognitive_profile["deep_reasoning"] = True


        # 🔹 Phase 3.2: World–Memory Binding
        binder = WorldMemoryBinder()
        emergent_concepts = binder.bind(world_state, emergent_concepts)

        
        # ===== Phase 3.5 : Emergent Concept Injection =====
        if cognitive_profile.get("use_emergent_concepts") and emergent_concepts:
            mutated_question = (
                mutated_question
                + " "
                + " ".join(emergent_concepts[:3])
            )

        #=================================================
        
        
         
         
        #===============================================

        logging.info(f"Emergent Concepts: {emergent_concepts}")
        
        max_docs = cognitive_profile["max_docs"]
        context = "\n\n".join(doc.page_content for doc in docs[:max_docs])

        res = chain.invoke({
            "context": context,
            "question": mutated_question
        })
        

    
    elif final_route == "reasoning":
        if cognitive_profile.get("deep_reasoning",False):
            reasoning_prompt = f"""
You are an advanced reasoning engine.
Think step by step.
Challenge assumptions.
Use indirect logic if needed.

Question:
{question}
"""
        else:
            reasoning_prompt = f"""
Answer briefly and clearly.

Question:
{question}
"""

        res = llm_generate(reasoning_prompt)
    
    if 'res' not in locals():
        res = llm_generate(question)

       
    
    # ================================
    # PHASE 5 — AGENCY (JARVIS ONLY)
    # ================================
    
    agency_result = None
    
    if SYSTEM_MODE == "jarvis" and intent_state in ["research", "execution"]:
    
        # ---- Phase 5.1 : Goal Formation ----
        goal_engine = GoalFormationEngine()
        goal = goal_engine.infer(
            question=question,
            intent=intent_state,
            world_state=world_state
        )
    
        # ---- Phase 5.2 : Agency Safety ----
        agency_safety = AgencySafetyEngine()
        agency_check = agency_safety.evaluate(
            goal=goal,
            mode=SYSTEM_MODE
        )
    
        if agency_check["allow"]:
    
            # ---- Phase 5.3 : Plan Synthesis ----
            plan_engine = PlanSynthesisEngine()
            plan = plan_engine.build(
                goal=goal,
                world_state=world_state
            )
    
            # ---- Phase 5.4 : Action Selection ----
            action_engine = ActionSelectionEngine()
            actions = action_engine.select(
                plan=plan,
                cognitive_profile=cognitive_profile
            )
    
            # ---- Phase 5.5 : Tool Invocation ----
            tool_engine = ToolInvocationEngine()
            action_results = tool_engine.invoke(actions)
    
            # ---- Phase 5.6 : Execution Monitoring ----
            monitor = ExecutionMonitorEngine()
            execution_report = monitor.evaluate(action_results)
    
            agency_result = {
                "goal": goal,
                "plan": plan,
                "actions": actions,
                "execution": execution_report
            }
    
        else:
            agency_result = {
                "blocked": True,
                "reason": agency_check["reason"]
            }
    
    # ===== Phase 5 → Response Injection =====
    if agency_result and not agency_result.get("blocked"):
        res = f"""
    {res}
    
    [AGENCY EXECUTION SUMMARY]
    Goal: {agency_result['goal']}
    Actions Taken: {agency_result['actions']}
    Execution Status: {agency_result['execution']}
    """
    # ---- Layer-5 Safety: Evidence Normalization ----
    docs = docs if 'docs' in locals() else []
    memory = memory if 'memory' in locals() else None


    # ================================
    # LAYER 5 — REASONING & SYNTHESIS
    # ================================
    
    evidence_engine = EvidenceAggregator()
    
    evidence = evidence_engine.collect(
        retrieved_docs=docs if 'docs' in locals() else None,
        memory=memory if 'memory' in locals() else None,
        reasoning_output=res,
        world_state=world_state,
        agency_result=agency_result
    )
    
    conflict_engine = ContradictionDetector()
    conflicts = conflict_engine.detect(evidence)
    
    trust_engine = SourceTrustScorer()
    trust_scores = trust_engine.score(evidence, cognitive_profile,route=final_route)
    
    synthesis_engine = FinalSynthesisEngine()
    res, judgement = synthesis_engine.synthesize(
        evidence,
        conflicts,
        trust_scores,
        intent_state,
        cognitive_profile
    )
    
    logging.info(f"[Layer-5 Judgement] → {judgement}")
    
    

    # ===== Phase 2.3 : Meta-Cognition =====
    # NOTE:
    # This is LIGHT meta-cognition.
    # - Single-pass evaluation
    # - Max ONE retry
    # - No recursive self-reflection
    # - No world-model or self-model awareness
    # This block MUST NOT be converted into a loop.
    # Meta retry is intentionally limited to ONE iteration
    # to avoid self-doubt loops in future phases.
    

    meta_engine = MetaCognitionEngine()
    meta = meta_engine.evaluate(res, SYSTEM_MODE)
    cognitive_profile["confidence"] = meta.get("confidence", cognitive_profile.get("confidence", 0.6))
    world_state["cognitive_confidence"] = cognitive_profile["confidence"]

    logging.info(f"[Meta-Cognition] → {meta}")
    
    if meta["retry"]:
        retry_prompt = f"""
    Re-evaluate your previous answer.
    Improve clarity, logic, and completeness.
    
    Original Question:
    {question}
    
    Previous Answer:
    {res}
    """
        res = llm_generate(retry_prompt)

    cognitive_profile["confidence"] *= 0.95
    
 
        
        # ================================
    # PHASE 6 — SELF TRAINING (JARVIS)
    # ================================
    
    if SYSTEM_MODE == "jarvis":
    
        # ---- Phase 6A : Alignment Fine-tuning ----
        aligner = AlignmentFineTuner()
        alignment_report = aligner.evaluate(
            question=question,
            answer=res,
            meta=meta,
            agency_result=agency_result,
            mode=SYSTEM_MODE
        )
    
        # ---- Phase 6B : Knowledge Fine-tuning ----
        tuner = KnowledgeFineTuner()
        knowledge_patch = tuner.update(
            question=question,
            world_state=world_state,
            cognitive_profile=cognitive_profile,
            alignment_report=alignment_report
        )
      

        
    # ============ LAYER 4: MEMORY GRAPH ================================
    # #👉 Dekho kya ho raha hai:

    #   Question → memory activate
      
    #   Memory → retrieval guide
      
    #   Retrieval → intelligent ho gaya
      
    #   Ye ChatGPT-style hybrid behavior hai

    # =========================
    # PHASE 2: IMPLICIT MEMORY EXECUTION
    # =========================

    # docs, emergent_concepts = implicit_memory_retrieval(
    #     vector_db,
    #     question,
    #     k=12
    # )

    # logging.info(f"Emergent Concepts (implicit): {emergent_concepts}")

    

    
    

    # # expanded_query = expand_query(llm, question)
    # # logging.info(f"Expanded Query: {expanded_query}")
    
    # # docs = retrieve_docs_expanded(vector_db, expanded_query)


    
    # # 3️⃣ Build context
    # context = "\n\n".join(doc.page_content for doc in docs[:6])

    
    # # 4️⃣ Final answer
    # res = chain.invoke({
    #     "context": context,
    #     "question": question
    # })

    



    # ===== Phase 2.6 : Response Assembly =====
    assembler = ResponseAssemblyEngine()
    final_response = assembler.assemble(res, response_strategy)

    # print("Response:")
    # print(final_response)
    #------------------------------


    

    # ──────────────── Layer 8: SAVE conversation (Important) ────────────────
    await memory_layer.update_conversation(
        conversation_id=conversation_id,
        question=question,
        answer=final_response,
        tier=SYSTEM_MODE,
        project_context="Vimana Project" if SYSTEM_MODE == "jarvis" else None
    )

        # Layer 4 Graph Sync - Concepts & Relations permanently save
    await memory_graph_full.sync_to_memory_graph(
        question=question,
        answer=final_response,
        tier=SYSTEM_MODE
    )

    # Optional debug
    project_ctx = await memory_layer.get_project_context(conversation_id)
    if project_ctx and SYSTEM_MODE == "jarvis":
        logging.info(f"[Jarvis Project Reminder] {project_ctx}")

   
    #---------------------




     # ===== Phase 2.8 : Trace Logging (Jarvis / Debug only) =====
    if SYSTEM_MODE == "jarvis":
        tracer = TraceLogger()
        tracer.log({
            "question": question,
            "route": final_route,
            "intent": intent_state,
            "cognitive_profile": cognitive_profile,
            "response_strategy": response_strategy,
            "meta_cognition": meta,
            "final_length": len(final_response.split())
        })
#------------------------------------------------------------
    # =========================
    # SECTION 7 — FINAL OUTPUT GUARD
    # =========================
    boundary_guard = OutputBoundaryGuard()
    final_response = boundary_guard.enforce(
        answer=final_response,
        mode=SYSTEM_MODE,
        intent=intent_state,
        cognitive_profile=cognitive_profile
    )
     
    # =========================
    # SECTION 8 — GOVERNANCE LAYER
    # =========================
    governor = DeploymentGovernor()
    final_response = governor.apply(
        mode=SYSTEM_MODE,
        intent=intent_state,
        response=final_response
    )

    print("\nFinal Response:")
    print(final_response)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
