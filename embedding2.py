import os
import glob
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate import WeaviateVectorStore
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# === Import splitter จาก doc_chunk_edit2.py ===
from doc_chunk_edit2 import AcademicDocumentSplitter

load_dotenv()

weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
data_folder = "E:/workspace/langchain-study/data" 
collection_name = "LangchainStudy"

weaviate_client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
    auth_credentials=Auth.api_key(weaviate_api_key)
)

if weaviate_client.is_ready():
    print("✅ เชื่อมต่อ Weaviate สำเร็จ!")
else:
    print("❌ ไม่สามารถเชื่อมต่อ Weaviate ได้! กรุณาตรวจสอบว่า Docker Container ทำงานอยู่หรือไม่")
    exit()

embeddings = OllamaEmbeddings(model="bge-m3")

# หาไฟล์ .txt ทั้งหมด
text_files = set()
for pattern in ["*.txt", "*.TXT"]:
    files = glob.glob(os.path.join(data_folder, pattern))
    for file in files:
        if os.path.exists(file):
            text_files.add(file)

text_files = list(text_files)

if not text_files:
    print("❌ ไม่พบไฟล์ .txt ในโฟลเดอร์")
    weaviate_client.close()
    exit()

print(f"📁 พบไฟล์ {len(text_files)} ไฟล์")

# === ใช้ AcademicDocumentSplitter ===
splitter = AcademicDocumentSplitter()

total_chunks = 0
total_chars = 0

for file_path in sorted(text_files):
    print(f"\n🔄 กำลังประมวลผล: {os.path.basename(file_path)}")
    try:
        content = splitter.read_file(file_path)
        if not content.strip():
            print(f"⚠️  ไฟล์ {file_path} ว่างหรืออ่านไม่ได้")
            continue

        chunks = splitter.split_document(content, source_file=file_path)

        # สร้าง Document objects
        docs = [
            Document(
                page_content=chunk["content"],
                metadata={k: v for k, v in chunk.items() if k != "content"}
            )
            for chunk in chunks
        ]

        print(f"   📝 ได้ {len(docs)} chunks จากไฟล์ {os.path.basename(file_path)}")

        # === push ทีละไฟล์เข้า Weaviate ===
        db = WeaviateVectorStore.from_documents(
            docs,
            embeddings,
            client=weaviate_client,
            index_name=collection_name,
            text_key="content"
        )

        total_chunks += len(docs)
        total_chars += sum(len(doc.page_content) for doc in docs)

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดกับไฟล์ {file_path}: {e}")

# === ตรวจสอบผลลัพธ์ ===
collection = weaviate_client.collections.get(collection_name)
total_objects = collection.aggregate.over_all(total_count=True).total_count

print(f"\n📊 สรุปทั้งหมด:")
print(f"  - ไฟล์: {len(text_files)}")
print(f"  - Chunks: {total_chunks}")
print(f"  - ขนาดรวม: {total_chars:,} ตัวอักษร")
print(f"📈 จำนวนข้อมูลใน collection: {total_objects}")

weaviate_client.close()