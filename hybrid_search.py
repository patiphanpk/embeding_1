import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.query import HybridFusion
from langchain_ollama import OllamaEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
collection_name = "LangchainStudy"
# print(weaviate.__version__)

weaviate_client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
    auth_credentials=Auth.api_key(weaviate_api_key)
)

if not weaviate_client.is_ready():
    print("❌ ไม่สามารถเชื่อมต่อ Weaviate ได้! กรุณาตรวจสอบว่า Docker Container ทำงานอยู่หรือไม่")
    exit()

print("✅ เชื่อมต่อ Weaviate สำเร็จ!")

# model_name = "BAAI/bge-m3"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": True}
# hf = HuggingFaceEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )

# embeddings = hf
embeddings = OllamaEmbeddings(model="bge-m3")

try:
    db = WeaviateVectorStore(
        client=weaviate_client,
        index_name=collection_name,
        text_key="content",  
        embedding=embeddings
    )
    print("✅ ระบบ query พร้อมใช้งาน!")

except Exception as e:
    print(f"❌ ไม่สามารถเชื่อมต่อ collection '{collection_name}' ได้: {e}")
    print("💡 กรุณาตรวจสอบว่าได้ทำ embedding ไว้แล้วหรือยัง")
    weaviate_client.close()
    exit()

def print_collections():
    collection = weaviate_client.collections.use(collection_name)
    response = collection.query.fetch_objects(
        include_vector=True,
        limit=1,
        return_properties=["source_file", "section_type", "chunk_id"]
    )
    for o in response.objects:
        print(f"{o.uuid}\n")
        print(o.vector)
        print(f"{o.properties}\n\n")


def hybrid_search_RRF(query, k=5):

    collection = weaviate_client.collections.get(collection_name)
    query_vector = embeddings.embed_query(query)
    response = collection.query.hybrid(
        query=query,
        vector=query_vector,
        fusion_type=HybridFusion.RANKED,
        limit=k,
        return_metadata=['score']
    )
        
    results = []
    for obj in response.objects:
    # ดึง metadata จาก properties
        properties = obj.properties
        metadata = {
            'source_file': properties.get('source_file', 'ไม่ระบุ'),
            'full_path': properties.get('full_path', ''),
            'chunk_id': properties.get('chunk_id', ''),
            'section_type': properties.get('section_type', '')
        }
            
        results.append({
            'content': properties.get('content', ''),  # เปลี่ยนจาก 'text' เป็น 'content'
            'metadata': metadata,
            'score': obj.metadata.score if obj.metadata.score else 0
        })
    return results

def hybrid_search_alpha(query, alpha=0.5, k=5):
    collection = weaviate_client.collections.get(collection_name)
    query_vector = embeddings.embed_query(query)
    response = collection.query.hybrid(
        query=query,
        vector=query_vector,
        alpha=alpha,  # ใช้น้ำหนักระหว่าง dense กับ sparse
        limit=k,
        return_metadata=['score']
    )

    results = []
    for obj in response.objects:
        properties = obj.properties
        metadata = {
            'source_file': properties.get('source_file', 'ไม่ระบุ'),
            'full_path': properties.get('full_path', ''),
            'chunk_id': properties.get('chunk_id', ''),
            'section_type': properties.get('section_type', '')
        }

        results.append({
            'content': properties.get('content', ''),
            'metadata': metadata,
            'score': obj.metadata.score if obj.metadata.score else 0
        })
    return results

# # ฟังก์ชันแสดงผลลัพธ์
def display_results(results, search_type, query):
    """แสดงผลลัพธ์การค้นหา"""
    print(f"\n{'='*20} {search_type} {'='*20}")
    print(f"🔍 คำค้นหา: '{query}'")
    
    if not results:
        print("❌ ไม่พบผลลัพธ์")
        return
    
    print(f"📄 พบผลลัพธ์ {len(results)} รายการ:")
    
    for i, result in enumerate(results, 1):
        source_file = result['metadata'].get('source_file', 'ไม่ระบุ')
        chunk_id = result['metadata'].get('chunk_id', '')
        section_type = result['metadata'].get('section_type', '')
        score = result.get('score', 0)
        content = result['content']
        
        print(f"\n📋 ผลลัพธ์ที่ {i}")
        print(f"   ไฟล์: {source_file}")
        if chunk_id:
            print(f"   Chunk: {chunk_id}")
        if section_type:
            print(f"   Section: {section_type}")
        print(f"   Score: {score:.4f}")
        print("-" * 50)
        print(content)
        print("-" * 50)

def compare_hybrid_scores(query, alpha=0.5, k=5):
    print(f"\n🔍 เปรียบเทียบ Hybrid Search: Alpha vs RRF")
    print(f"คำค้นหา: '{query}'\n{'='*60}")

    results_alpha = hybrid_search_alpha(query, alpha=alpha, k=k)
    results_rrf = hybrid_search_RRF(query, k=k)

    print(f"\n{'ผลลัพธ์จาก Hybrid Alpha':^60}")
    print("-"*60)
    for i, r in enumerate(results_alpha, 1):
        content_preview = (r['content'][:100] + "...") if len(r['content']) > 100 else r['content']
        print(f"{i}. Score: {r['score']:.4f} | Source: {r['metadata'].get('source_file', 'ไม่ระบุ')}")
        print(f"   📝 {content_preview}")

    print(f"\n{'ผลลัพธ์จาก Hybrid RRF':^60}")
    print("-"*60)
    for i, r in enumerate(results_rrf, 1):
        content_preview = (r['content'][:100] + "...") if len(r['content']) > 100 else r['content']
        print(f"{i}. Score: {r['score']:.4f} | Source: {r['metadata'].get('source_file', 'ไม่ระบุ')}")
        print(f"   📝 {content_preview}")

print_collections()
query = "ในช่วงฤดูหนาวของประเทศไทยมักเกิดปัญหาฝุ่น PM2.5 สะสมในอากาศ เนื่องจากสภาพอากาศที่นิ่งและการเผาในที่โล่ง หากต้องการลดผลกระทบต่อสุขภาพของประชาชน ควรมีมาตรการเชิงรุกจากทั้งภาครัฐและประชาชน เช่น การตรวจสอบคุณภาพอากาศรายวัน การลดการใช้รถยนต์ส่วนตัว และการส่งเสริมระบบขนส่งสาธารณะพลังงานสะอาด"
compare_hybrid_scores(query, alpha=0.5, k=5)

# ปิดการเชื่อมต่อ
weaviate_client.close()
print("🔒 ปิดการเชื่อมต่อแล้ว")