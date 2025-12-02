import os
import glob
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate import WeaviateVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# โหลดค่าจาก .env
load_dotenv()

# ตั้งค่า
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
data_folder = "E:/workspace/langchain-study/data"  # เปลี่ยนเป็น path ของคุณ
collection_name = "LangchainStudy"

# เชื่อมต่อ Weaviate
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

#ของ HuggingFace

# model_name = "BAAI/bge-m3"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": True}
# hf = HuggingFaceEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )

# embeddings = hf

# สร้าง embeddings model
embeddings = OllamaEmbeddings(model="bge-m3")

# หาไฟล์ .txt ทั้งหมดในโฟลเดอร์ (แก้ไขปัญหาไฟล์ซ้ำ)
text_files = set()  # ใช้ set เพื่อป้องกันไฟล์ซ้ำ

# ใช้ case-insensitive pattern
for pattern in ["*.txt", "*.TXT"]:
    files = glob.glob(os.path.join(data_folder, pattern))
    for file in files:
        # normalize path เพื่อป้องกันไฟล์ซ้ำ
        normalized_path = os.path.normpath(file.lower())
        if os.path.exists(file):  # ตรวจสอบว่าไฟล์มีจริง
            text_files.add(file)

# แปลงกลับเป็น list
text_files = list(text_files)

if not text_files:
    print("❌ ไม่พบไฟล์ .txt ในโฟลเดอร์")
    weaviate_client.close()
    exit()

print(f"📁 พบไฟล์ {len(text_files)} ไฟล์:")
for file in sorted(text_files):  # เรียงลำดับเพื่อให้ดูง่าย
    print(f"  - {os.path.basename(file)} ({os.path.getsize(file)} bytes)")

# ตรวจสอบ collection เก่า และลบถ้าต้องการเริ่มใหม่
try:
    if weaviate_client.collections.exists(collection_name):
        print(f"⚠️  Collection '{collection_name}' มีอยู่แล้ว")
        response = input("ต้องการลบและสร้างใหม่หรือไม่? (y/N): ")
        if response.lower() == 'y':
            weaviate_client.collections.delete(collection_name)
            print(f"🗑️  ลบ Collection '{collection_name}' เรียบร้อย")
        else:
            print("❌ ยกเลิกการดำเนินการ")
            weaviate_client.close()
            exit()
except Exception as e:
    print(f"⚠️  ไม่สามารถตรวจสอบ collection ได้: {e}")

# ประมวลผลทุกไฟล์
all_chunks = []
processed_content = set()  # เก็บ hash ของเนื้อหาเพื่อป้องกันข้อมูลซ้ำ
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

for file_path in text_files:
    try:
        print(f"\n🔄 กำลังประมวลผล: {os.path.basename(file_path)}")
        
        # โหลดเอกสาร
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        
        # ตรวจสอบเนื้อหาซ้ำ
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash in processed_content:
                print(f"   ⚠️  พบเนื้อหาซ้ำในไฟล์ {os.path.basename(file_path)} - ข้าม")
                continue
            
            processed_content.add(content_hash)

         
        
        # แบ่งข้อความ
        chunks = text_splitter.split_documents(documents)
        
        # กรองข้อมูลซ้ำในระดับ chunk
        unique_chunks = []
        chunk_contents = set()
        
        for chunk in chunks:
            chunk_hash = hash(chunk.page_content)
            if chunk_hash not in chunk_contents:
                chunk_contents.add(chunk_hash)
                unique_chunks.append(chunk)
        
        all_chunks.extend(unique_chunks)
        
        if len(chunks) != len(unique_chunks):
            print(f"   📝 แบ่งได้ {len(chunks)} chunks (หลังกรองซ้ำ: {len(unique_chunks)} chunks)")
        else:
            print(f"   📝 แบ่งได้ {len(unique_chunks)} chunks")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดกับไฟล์ {file_path}: {e}")

if not all_chunks:
    print("❌ ไม่มีข้อมูลที่จะทำ embedding")
    weaviate_client.close()
    exit()

print(f"\n📊 สรุป:")
print(f"  - ไฟล์ทั้งหมด: {len(text_files)} ไฟล์")
print(f"  - Chunks ทั้งหมด: {len(all_chunks)} chunks")
print(f"  - ขนาดเนื้อหารวม: {sum(len(chunk.page_content) for chunk in all_chunks):,} ตัวอักษร")

print(f"\n🚀 กำลังสร้าง embeddings สำหรับ {len(all_chunks)} chunks...")

# สร้าง vector store
try:
    db = WeaviateVectorStore.from_documents(
        all_chunks,
        embeddings,
        client=weaviate_client,
        index_name=collection_name,
        text_key="text"
    )
    print("✅ สร้าง embeddings เสร็จสิ้น!")
    
    # ตรวจสอบจำนวนข้อมูลใน collection
    collection = weaviate_client.collections.get(collection_name)
    total_objects = collection.aggregate.over_all(total_count=True).total_count
    print(f"📈 จำนวนข้อมูลใน collection: {total_objects}")
    weaviate_client.close()
    
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการสร้าง embeddings: {e}")
    weaviate_client.close()
    exit()