# catalog.py
import json
import re
import httpx
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
COLLECTION_NAME = "shl_catalog"


def fetch_catalog() -> list[dict]:
    """SHL catalog fetch karta hai — control characters clean karke"""
    response = httpx.get(
        CATALOG_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
        follow_redirects=True,
    )
    response.raise_for_status()

    # Yahi fix hai — invalid control characters remove karo JSON parse se pehle
    raw = response.content.decode("utf-8", errors="replace")
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)

    data = json.loads(clean, strict=False)

    # Structure handle karo
    if isinstance(data, list):
        products = data
    elif isinstance(data, dict):
        # Pehli list value lo jo bhi key ho
        products = next(
            (v for v in data.values() if isinstance(v, list)), []
        )
    else:
        products = []

    print(f"Catalog se {len(products)} products mile.")
    if products:
        print(f"Sample product keys: {list(products[0].keys())}")
    return products


# def build_document(product: dict) -> Document:
#     """Product dict se LangChain Document banata hai"""

#     # Catalog ke actual field names — common variations handle karo
#     def get(keys, default=""):
#         for k in keys:
#             if product.get(k):
#                 return str(product[k])
#         return default

#     name = get(["name", "title", "product_name"])
#     description = get(["description", "desc", "overview", "about"])
#     test_type = get(["test_type", "type", "category", "test_types"])
#     url = get(["url", "link", "product_url", "catalog_url"])
#     duration = get(["duration", "time", "assessment_time", "timing"])
#     languages = product.get("languages", product.get("language", []))

#     if isinstance(languages, list):
#         lang_str = ", ".join(str(l) for l in languages[:5])
#     else:
#         lang_str = str(languages)

#     page_content = (
#         f"Assessment Name: {name}\n"
#         f"Test Type: {test_type}\n"
#         f"Description: {description}\n"
#         f"Duration: {duration}\n"
#         f"Languages: {lang_str}"
#     )

#     return Document(
#         page_content=page_content,
#         metadata={
#             "name": name,
#             "url": url,
#             "test_type": test_type,
#             "duration": duration,
#         },
#     )

def build_document(product: dict) -> Document:
    name = product.get("name", "").replace("\n", " ").strip()
    description = product.get("description", "")
    test_type = ",".join(product.get("keys", []))  # keys = ["K"], ["P","A"] etc
    url = product.get("link", "")
    duration = product.get("duration", "")
    languages = product.get("languages", [])
    job_levels = ",".join(product.get("job_levels", []))

    if isinstance(languages, list):
        lang_str = ", ".join(str(l) for l in languages[:5])
    else:
        lang_str = str(languages)

    page_content = (
        f"Assessment Name: {name}\n"
        f"Test Type: {test_type}\n"
        f"Description: {description}\n"
        f"Duration: {duration}\n"
        f"Languages: {lang_str}\n"
        f"Job Levels: {job_levels}"
    )

    return Document(
        page_content=page_content,
        metadata={
            "name": name,
            "url": url,
            "test_type": test_type,
            "duration": str(duration),
        },
    )


def get_vectorstore(mistral_api_key: str) -> Chroma:
    """ChromaDB vectorstore return karta hai"""
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=mistral_api_key,
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    if vectorstore._collection.count() == 0:
        print("Pehli baar catalog load ho raha hai ChromaDB mein...")
        products = fetch_catalog()

        if not products:
            raise RuntimeError("Catalog empty aaya — URL check karo.")

        documents = [build_document(p) for p in products]
        vectorstore.add_documents(documents)
        print(f"{len(documents)} assessments store ho gaye ChromaDB mein!")
    else:
        print(f"ChromaDB ready — {vectorstore._collection.count()} assessments loaded.")

    return vectorstore