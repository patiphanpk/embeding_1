import os
import glob
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings

# === Import splitter จาก doc_chunk_edit2.py ===
try:
    from doc_chunk_edit2 import AcademicDocumentSplitter
except ImportError:
    print("❌ ไม่พบไฟล์ doc_chunk_edit2.py หรือคลาส AcademicDocumentSplitter")
    exit()

# --- การตั้งค่า (Configuration) ---
load_dotenv()

WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
DATA_FOLDER = "E:/workspace/langchain-study/data" 
BASE_COLLECTION_NAME = "LangchainStudy"

# รายการโมเดล BGE จาก Ollama ที่ต้องการทดสอบ
BGE_MODELS = [
    "bge-m3",
    "bge-large",
    "mxbai-embed-large",
    "snowflake-arctic-embed" 
]

def load_and_split_documents(folder_path: str) -> list[Document]:
    """
    โหลดและแบ่งไฟล์ด้วย AcademicDocumentSplitter ที่ถูกต้อง
    """
    print(f"\n📁 กำลังอ่านและแบ่งไฟล์จากโฟลเดอร์ '{folder_path}'...")
    if not os.path.exists(folder_path):
        print(f"❌ ไม่พบโฟลเดอร์ '{folder_path}'")
        return []

    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
    all_langchain_docs = []
    
    splitter = AcademicDocumentSplitter(
        max_chunk_size=1000, 
        overlap_size=0,
        min_chunk_size=100,
        quality_threshold=0.3
    )

    for file_path in file_paths:
        try:
            filename = os.path.basename(file_path)
            content = splitter.read_file(file_path)
            
            if not content:
                print(f"  - ไม่สามารถอ่านเนื้อหาไฟล์: {filename}")
                continue

            chunk_dictionaries = splitter.split_document(content, source_file=filename)

            temp_docs = []
            for chunk_dict in chunk_dictionaries:
                page_content = chunk_dict.pop('content', '') 
                metadata = chunk_dict 
                new_doc = Document(page_content=page_content, metadata=metadata)
                temp_docs.append(new_doc)

            all_langchain_docs.extend(temp_docs)
            print(f"  - ไฟล์ '{filename}' ถูกแบ่งเป็น {len(temp_docs)} chunks")
            
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการอ่านหรือแบ่งไฟล์ {os.path.basename(file_path)}: {e}")

    print(f"✅ การแบ่งเอกสารเสร็จสิ้น, ได้ chunks ทั้งหมด {len(all_langchain_docs)} ชิ้น")
    return all_langchain_docs


def create_hybrid_collection(client: weaviate.WeaviateClient, collection_name: str, vector_dim: int = 1024):
    """
    สร้าง Collection ที่รองรับ Hybrid Search (Dense Vector + BM25 Sparse)
    """
    if client.collections.exists(collection_name):
        print(f"พบ Collection เก่า '{collection_name}', กำลังลบ...")
        client.collections.delete(collection_name)
    
    print(f"กำลังสร้าง Collection '{collection_name}' แบบ Hybrid Search...")
    
    # สร้าง Collection พร้อม BM25 configuration
    client.collections.create(
        name=collection_name,
        vectorizer_config=None,  # ใช้ vector ที่เราสร้างเอง
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source_file", data_type=DataType.TEXT),
            Property(name="main_topics", data_type=DataType.TEXT_ARRAY),  # เพิ่ม main_topics
        ],
        # เปิดใช้งาน BM25 สำหรับ keyword search
        inverted_index_config=Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2
        )
    )
    print("✅ สร้าง Collection สำเร็จ (รองรับ Dense Vector + BM25)")


def ingest_documents_with_embeddings(
    client: weaviate.WeaviateClient, 
    collection_name: str, 
    documents: list[Document],
    embeddings_model: OllamaEmbeddings
):
    """
    เพิ่มข้อมูลพร้อม embeddings เข้า Weaviate
    """
    print(f"\n📥 กำลังทำ Embedding และ Ingest ข้อมูล {len(documents)} chunks...")
    
    collection = client.collections.get(collection_name)
    
    # แบ่งการ embed เป็น batch เพื่อประสิทธิภาพ
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        texts = [doc.page_content for doc in batch_docs]
        
        # สร้าง embeddings แบบ batch
        print(f"  - กำลัง embed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")
        vectors = embeddings_model.embed_documents(texts)
        
        # เพิ่มข้อมูลเข้า Weaviate พร้อม vector
        with collection.batch.dynamic() as batch:
            for doc, vector in zip(batch_docs, vectors):
                # ดึง main_topics จาก metadata (ถ้ามี)
                main_topics = doc.metadata.get('main_topics', [])
                # ถ้าเป็น string ให้แปลงเป็น list
                if isinstance(main_topics, str):
                    main_topics = [main_topics] if main_topics else []
                
                batch.add_object(
                    properties={
                        "text": doc.page_content,
                        "source_file": doc.metadata.get('source_file', 'N/A'),
                        "main_topics": main_topics,  # เพิ่ม main_topics
                    },
                    vector=vector
                )
    
    print("✅ ข้อมูลทั้งหมดถูกเพิ่มเข้า Weaviate เรียบร้อยแล้ว")


def hybrid_search(
    client: weaviate.WeaviateClient,
    collection_name: str,
    query: str,
    embeddings_model: OllamaEmbeddings,
    alpha: float = 0.5,
    limit: int = 3
):
    """
    ค้นหาแบบ Hybrid (Dense + Sparse)
    
    Args:
        alpha: 0.0 = pure BM25, 1.0 = pure vector, 0.5 = balanced hybrid
    """
    collection = client.collections.get(collection_name)
    
    # สร้าง query vector
    print(f"  🔍 กำลังสร้าง query embedding...")
    query_vector = embeddings_model.embed_query(query)
    
    # ทำ Hybrid Search
    print(f"  🔍 กำลังค้นหาแบบ Hybrid (alpha={alpha})...")
    response = collection.query.hybrid(
        query=query,
        vector=query_vector,
        alpha=alpha,  # 0.5 = ให้น้ำหนักเท่ากันระหว่าง vector และ keyword
        limit=limit,
        return_metadata=MetadataQuery(score=True, explain_score=True)
    )
    
    return response.objects


def run_test_for_model(client: weaviate.WeaviateClient, model_name: str, documents: list[Document]):
    if not documents:
        print("ไม่มีเอกสารให้ประมวลผล, ข้ามการทดสอบสำหรับโมเดลนี้")
        return

    collection_name = f"{BASE_COLLECTION_NAME}_{model_name.replace('-', '_').replace('.', '_')}"
    print(f"\n{'='*70}")
    print(f"🧪 เริ่มการทดสอบสำหรับโมเดล: {model_name}")
    print(f"📦 Collection: {collection_name}")
    print(f"{'='*70}")

    # สร้าง embeddings model
    embeddings = OllamaEmbeddings(model=model_name)
    
    # สร้าง Collection แบบ Hybrid
    create_hybrid_collection(client, collection_name)
    
    # เพิ่มข้อมูลพร้อม embeddings
    ingest_documents_with_embeddings(client, collection_name, documents, embeddings)
    
    # ทดสอบค้นหา
    query = "สถานการณ์ฝุ่น PM2.5 เป็นอย่างไร"
    print(f"\n{'='*70}")
    print(f"🔍 ทดสอบ Hybrid Search: \"{query}\"")
    print(f"{'='*70}")
    
    try:
        # ทดสอบ 3 แบบ
        test_configs = [
            (0.0, "Pure BM25 (Keyword Only)"),
            (0.5, "Hybrid (Balanced)"),
            (1.0, "Pure Vector (Semantic Only)")
        ]
        
        for alpha, description in test_configs:
            print(f"\n\n📊 {description} (alpha={alpha})")
            print("─" * 70)
            
            results = hybrid_search(client, collection_name, query, embeddings, alpha=alpha, limit=3)
            
            if not results:
                print("❌ ไม่พบผลการค้นหา")
                continue
            
            for i, obj in enumerate(results, 1):
                props = obj.properties
                score = obj.metadata.score if obj.metadata.score else 0
                explain = obj.metadata.explain_score if hasattr(obj.metadata, 'explain_score') else None
                
                print(f"\n  [{i}] 📄 ไฟล์: {props.get('source_file', 'N/A')}")
                print(f"      ✨ Hybrid Score: {score:.4f}")
                
                # แสดง explain_score ถ้ามี
                if explain:
                    print(f"      🔬 Score Explain: {explain}")
                
                print(f"      🏷️  Topics: {', '.join(props.get('main_topics', [])) or 'N/A'}")
                print(f"      📖 เนื้อหา: {props.get('text', '')[:200].replace(os.linesep, ' ')}...")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการค้นหา: {e}")
        import traceback
        traceback.print_exc()


def main():
    weaviate_client = None
    try:
        weaviate_client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
        )
        if not weaviate_client.is_ready():
            raise ConnectionError("Weaviate is not ready.")
        print("✅ เชื่อมต่อ Weaviate สำเร็จ!")
    except Exception as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ Weaviate ได้! : {e}")
        exit()

    try:
        documents = load_and_split_documents(DATA_FOLDER)
        if not documents:
            print("❌ ไม่มีเอกสารให้ประมวลผล")
            return

        while True:
            print(f"\n{'='*70}")
            print("🚀 โปรแกรมทดสอบ Hybrid Search (Dense Vector + BM25 Sparse)")
            print(f"{'='*70}")
            for i, model in enumerate(BGE_MODELS):
                print(f"  [{i+1}] ทดสอบโมเดล: {model}")
            print("  [0] ออกจากโปรแกรม")
            print("="*70)
            
            try:
                choice_input = input("\nเลือกโมเดล (ตัวเลข): ").strip()
                
                if not choice_input:
                    print("❌ กรุณาป้อนตัวเลข")
                    continue
                    
                choice = int(choice_input)
                
                if choice == 0:
                    print("👋 ออกจากโปรแกรม")
                    break
                elif 1 <= choice <= len(BGE_MODELS):
                    selected_model = BGE_MODELS[choice - 1]
                    run_test_for_model(weaviate_client, selected_model, documents)
                else:
                    print("❌ ตัวเลือกไม่ถูกต้อง กรุณาเลือกตัวเลขในช่วงที่กำหนด")
            except ValueError:
                print("❌ กรุณาป้อนเป็นตัวเลขเท่านั้น")
            except KeyboardInterrupt:
                print("\n\n👋 ออกจากโปรแกรม (Ctrl+C)")
                break
    finally:
        if weaviate_client:
            weaviate_client.close()
            print("\n🔌 ปิดการเชื่อมต่อ Weaviate เรียบร้อยแล้ว")


if __name__ == "__main__":
    main()