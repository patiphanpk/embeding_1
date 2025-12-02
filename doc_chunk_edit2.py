# ผลลัพธ์ที่แตกต่าง
# ไฟล์แรก 

# Chunks มากกว่า (รวมทั้งคุณภาพต่ำ)
# Overlap อาจตัดกลางประโยค
# ขนาด chunks ไม่สม่ำเสมอ

# ไฟล์ที่สอง 

# Chunks คุณภาพสูงกว่า (ผ่านการกรอง)
# Overlap ที่จุดเหมาะสม (จบประโยค/คำ)
# ขนาดและคุณภาพสม่ำเสมอกว่า

# ที่เพิ่มมา 
#  Overlap Handling


# Splitting Strategies
# ไฟล์แรก: 4 รูปแบบ
# หัวข้อย่อย
# รายการไทย/อาหรับ
# ย่อหน้า
# recursive fallback

# ไฟล์ที่สอง: 8 รูปแบบ
# หัวข้อหลัก
# หัวข้อรอง
# หัวข้อย่อย
# รายการเลขไทย
# รายการเลขอารบิค
# รายการอักษรไทย
# ย่อหน้า
# ประโยค



# การแบ่งระดับ (Levels) ในตัวอย่างนี้

# Level 1: Document Title / Main Header

# "document_title": "มติคณะกรรมการสภาวิชาการ"
# → ระบุว่าเป็นมติสภาวิชาการ

# Level 2: Section Type / หมวดของเอกสาร

# เช่น "section_type": "resolution", "appointment", "curriculum"
# → ช่วยจัดกลุ่มว่า chunk นั้นเป็นมติ, การแต่งตั้ง, หรือรายละเอียดหลักสูตร

# Level 3: Header Hierarchy (header_info)

# main_headers: เช่น # หลักสูตรวิทยาศาสตรมหาบัณฑิต สาขาวิชาการจัดการความปลอดภัยอาหาร (chunk 5)

# sub_headers: (ยังไม่มีในไฟล์นี้ แต่รองรับได้)

# sub_sub_headers: เช่น ### เรื่อง ขอความเห็นชอบการปฏิบัติ... (chunk 0)

# "has_hierarchy": true/false บอกว่ามีโครงสร้าง header ชัดเจนหรือไม่

# Level 4: Content Body / เนื้อหา

# เป็นเนื้อหาหลัก เช่น รายละเอียดการประชุม, ข้อ 1.1, 1.2, หรือคำอธิบายหลักสูตร

# Level 5: Table / รายละเอียดเชิงลึก

# เช่น รายละเอียดรหัสวิชา, รายชื่ออาจารย์, หน่วยกิต, เหตุผลความจำเป็น

# ระบบ detect ผ่าน contains_table: true
import re
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import hashlib

class AcademicDocumentSplitter:
    """
    Advanced Splitter เฉพาะสำหรับเอกสารวิชาการ ภาษาไทย
    Features:
    - Complete header hierarchy support (# ## ###)
    - Smart structure detection
    - Context preservation 
    - Advanced metadata extraction
    - Overlap handling for better context
    - Quality scoring
    """
    
    def __init__(self, 
                 max_chunk_size: int = 1500,
                 overlap_size: int = 150,
                 preserve_tables: bool = True,
                 preserve_sections: bool = True,
                 min_chunk_size: int = 100,
                 quality_threshold: float = 0.5):
        
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.preserve_tables = preserve_tables
        self.preserve_sections = preserve_sections
        self.quality_threshold = quality_threshold
        
        # รูปแบบการจดจำโครงสร้างเอกสารวิชาการ (Enhanced with complete headers)
        self.section_patterns = [
            (r'^#\s+(.+)$', 1, 'main_header'),           # หัวข้อหลัก
            (r'^##\s+(.+)$', 2, 'sub_header'),          # หัวข้อรอง
            (r'^###\s+(.+)$', 3, 'sub_sub_header'),     # หัวข้อย่อย
            (r'^[๑-๙]+[\)\.]\s*(.+)$', 4, 'thai_list'),  # รายการแบบไทย
            (r'^[1-9][0-9]*[\)\.]\s*(.+)$', 4, 'arabic_list'), # รายการแบบอาหรับ
            (r'^[ก-ฮ][\)\.]\s*(.+)$', 5, 'thai_alpha_list'), # รายการแบบ ก ข ค
            (r'^\([๑-๙]+\)\s*(.+)$', 5, 'thai_paren_list'), # (๑) (๒)
            (r'^\([1-9][0-9]*\)\s*(.+)$', 5, 'arabic_paren_list'), # (1) (2)
        ]
        
        # Enhanced split strategies with complete header support
        self.split_strategies = [
            (r'\n(?=#\s)', 'main_header'),               # หัวข้อหลัก
            (r'\n(?=##\s)', 'sub_header'),               # หัวข้อรอง
            (r'\n(?=###\s)', 'sub_sub_header'),          # หัวข้อย่อย
            (r'\n(?=[๑-๙]+[\)\.]\s)', 'thai_numbered'),  # รายการเลขไทย
            (r'\n(?=[1-9][0-9]*[\)\.]\s)', 'arabic_numbered'), # รายการเลขอาหรับ
            (r'\n(?=\([๑-๙]+\)\s)', 'thai_parentheses'), # รายการไทยในวงเล็บ
            (r'\n(?=\([1-9][0-9]*\)\s)', 'arabic_parentheses'), # รายการอาหรับในวงเล็บ
            (r'\n(?=[ก-ฮ][\)\.]\s)', 'thai_alpha'),      # รายการ ก ข ค
            (r'\n\n+', 'paragraphs'),                    # ย่อหน้า
            (r'(?<=\.)\s+(?=[A-Z])', 'sentences'),       # ประโยค
        ]
        
        # รูปแบบตาราง (Enhanced)
        self.table_patterns = [
            r'\|.*\|.*\|',               # Markdown table (3+ columns)
            r'<table>.*?</table>',       # HTML table
            r'┌.*┐',                     # ASCII box table
            r'╔.*╗',                     # Double line box table
            r'\+[-=]+\+',                # Simple ASCII table
        ]
        
        # รูปแบบรหัสวิชา (Enhanced)
        self.course_code_patterns = [
            r'[๐-๙]{8}',                # รหัสวิชา 8 หลักไทย
            r'[0-9]{8}',                # รหัสวิชา 8 หลักอาหรับ
            r'[๐-๙]{6}',                # รหัสวิชา 6 หลักไทย
            r'[0-9]{6}',                # รหัสวิชา 6 หลักอาหรับ
            r'[A-Z]{2,4}[0-9]{3,4}',    # รหัสวิชาแบบตะวันตก
        ]
        
        # คำสำคัญประเภทเอกสาร
        self.document_types = {
            'curriculum': ['หลักสูตร', 'รายวิชา', 'สาขาวิชา', 'ปริญญา'],
            'resolution': ['มติ', 'ที่ประชุม', 'เห็นชอบ', 'อนุมัติ'],
            'modification': ['ปรับปรุง', 'แก้ไข', 'เปลี่ยนแปลง', 'ปรับเปลี่ยน'],
            'rationale': ['เหตุผล', 'ความจำเป็น', 'ที่มาของ'],
            'table': ['ตาราง', 'รายการ', 'สรุป'],
            'appointment': ['แต่งตั้ง', 'อาจารย์', 'กรรมการ'],
            'report': ['รายงาน', 'ผลการ', 'สรุปผล'],
        }
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """ตั้งค่า logging"""
        # สร้าง logs directory ถ้าไม่มี
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / 'document_processing.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def read_file(self, file_path: Union[str, Path]) -> str:
        """อ่านไฟล์ข้อความด้วย encoding detection"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return ""
            
        encodings = ['utf-8', 'cp874', 'tis-620', 'utf-8-sig']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                self.logger.debug(f"Successfully read {file_path} with {encoding}")
                return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                self.logger.error(f"Error reading file {file_path}: {e}")
                continue
        
        # Last resort - ignore errors
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            self.logger.warning(f"Read {file_path} with error ignoring")
            return content
        except Exception as e:
            self.logger.error(f"Failed to read {file_path}: {e}")
            return ""
    
    def process_files_recursive(self, folder_path: Union[str, Path]) -> Dict[str, List[Dict]]:
        """
        ประมวลผลไฟล์แบบ recursive พร้อม progress tracking
        """
        results = {}
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            self.logger.error(f"Folder {folder_path} not found!")
            return results
        
        # หาไฟล์ .txt ทั้งหมดรวมโฟลเดอร์ย่อย
        txt_files = list(folder_path.rglob("*.txt"))
        
        if not txt_files:
            self.logger.warning(f"No .txt files found in {folder_path}")
            return results
        
        self.logger.info(f"Found {len(txt_files)} text files in {folder_path}")
        
        processed = 0
        failed = 0
        
        for file_path in txt_files:
            try:
                # สร้าง key ที่รวม path สัมพันธ์
                relative_path = file_path.relative_to(folder_path)
                key = str(relative_path)
                
                # Progress indicator
                processed += 1
                if processed % 10 == 0 or processed == len(txt_files):
                    print(f"Processing... {processed}/{len(txt_files)} ({processed/len(txt_files)*100:.1f}%)")
                
                # อ่านไฟล์
                content = self.read_file(file_path)
                
                if content.strip():
                    # แบ่งเอกสาร
                    chunks = self.split_document(content, source_file=key)
                    if chunks:
                        results[key] = chunks
                        self.logger.debug(f"Successfully processed {key}: {len(chunks)} chunks")
                    else:
                        self.logger.warning(f"No chunks generated for {key}")
                else:
                    self.logger.warning(f"Empty or unreadable file: {key}")
                    failed += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                failed += 1
                continue
        
        self.logger.info(f"Processing complete: {len(results)} files processed, {failed} failed")
        return results
    
    def split_document(self, text: str, source_file: str = "") -> List[Dict]:
        """
        แบ่งเอกสารด้วยอัลกอริทึมที่ปรับปรุง
        """
        if not text.strip():
            return []
        
        # 1. วิเคราะห์โครงสร้างเอกสาร
        structure = self._analyze_structure(text)
        
        # 2. แบ่งตามโครงสร้าง
        chunks = self._split_by_structure(text, structure)
        
        # 3. จัดการ overlap และรวม chunks เล็ก
        chunks = self._handle_overlaps_and_merge(chunks)
        
        # 4. เพิ่ม metadata และคำนวณคุณภาพ
        enriched_chunks = self._add_enhanced_metadata(chunks, structure, source_file)
        
        # 5. กรองตาม quality threshold
        quality_chunks = [chunk for chunk in enriched_chunks 
                         if chunk['quality_score'] >= self.quality_threshold]
        
        if not quality_chunks and enriched_chunks:
            # ถ้าไม่มี chunk ไหนผ่าน threshold ให้เอาที่ดีที่สุด
            quality_chunks = [max(enriched_chunks, key=lambda x: x['quality_score'])]
        
        return quality_chunks
    
    def _analyze_structure(self, text: str) -> Dict:
        """วิเคราะห์โครงสร้างเอกสารอย่างละเอียด"""
        
        structure = {
            'title': '',
            'sections': [],
            'tables': [],
            'lists': [],
            'course_codes': set(),
            'document_type': 'general',
            'header_hierarchy': {'main': [], 'sub': [], 'sub_sub': []},
            'metadata': {}
        }
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # หาหัวข้อตามรูปแบบต่างๆ
            for pattern, level, section_type in self.section_patterns:
                match = re.match(pattern, line)
                if match:
                    section_info = {
                        'title': match.group(1).strip(),
                        'level': level,
                        'type': section_type,
                        'start_line': i,
                        'content': '',
                        'line_text': line
                    }
                    structure['sections'].append(section_info)
                    
                    # จัดเก็บตาม hierarchy
                    if level == 1:
                        structure['header_hierarchy']['main'].append(section_info)
                    elif level == 2:
                        structure['header_hierarchy']['sub'].append(section_info)
                    elif level == 3:
                        structure['header_hierarchy']['sub_sub'].append(section_info)
                    
                    break
            
            # หาตาราง
            for pattern in self.table_patterns:
                if re.search(pattern, line):
                    structure['tables'].append({
                        'line': i,
                        'content': line,
                        'pattern': pattern
                    })
                    break
            
            # หารหัสวิชา
            for pattern in self.course_code_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    structure['course_codes'].add(match)
        
        # กำหนดชื่อเอกสาร (ลำดับความสำคัญ: main header > sub header > first section)
        if structure['header_hierarchy']['main']:
            structure['title'] = structure['header_hierarchy']['main'][0]['title']
        elif structure['header_hierarchy']['sub']:
            structure['title'] = structure['header_hierarchy']['sub'][0]['title']
        elif structure['sections']:
            structure['title'] = structure['sections'][0]['title']
        
        # ระบุประเภทเอกสาร
        structure['document_type'] = self._classify_document_type(text)
        
        return structure
    
    def _classify_document_type(self, text: str) -> str:
        """จำแนกประเภทเอกสาร"""
        text_lower = text.lower()
        type_scores = {}
        
        for doc_type, keywords in self.document_types.items():
            score = sum(text_lower.count(keyword.lower()) for keyword in keywords)
            if score > 0:
                type_scores[doc_type] = score
        
        return max(type_scores, key=type_scores.get) if type_scores else 'general'
    
    def _split_by_structure(self, text: str, structure: Dict) -> List[str]:
        """แบ่งข้อความตามโครงสร้างอย่างชาญฉลาด"""
        
        chunks = []
        sections = structure['sections']
        lines = text.split('\n')
        
        if not sections:
            return self._fallback_split(text)
        
        for i, section in enumerate(sections):
            start_line = section['start_line']
            
            # หาจุดสิ้นสุดของส่วนนี้
            if i + 1 < len(sections):
                end_line = sections[i + 1]['start_line']
            else:
                end_line = len(lines)
            
            # ดึงเนื้อหารวมบริบท
            content_lines = lines[start_line:end_line]
            content = '\n'.join(content_lines).strip()
            
            # ตรวจสอบขนาดและแบ่งตามความเหมาะสม
            if len(content) <= self.max_chunk_size:
                if len(content) >= self.min_chunk_size:
                    chunks.append(content)
            else:
                # แบ่งย่อยด้วยการรักษาบริบท
                sub_chunks = self._intelligent_split_section(content, section)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _intelligent_split_section(self, content: str, section_info: Dict) -> List[str]:
        """แบ่งส่วนใหญ่อย่างชาญฉลาดโดยรักษาบริบท"""
        
        # ลำดับความสำคัญในการแบ่ง (รวม header hierarchy)
        for pattern, strategy in self.split_strategies:
            parts = re.split(pattern, content)
            
            if len(parts) > 1:
                result = []
                current_chunk = ""
                
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    
                    potential_chunk = current_chunk + ('\n' if current_chunk else '') + part
                    
                    if len(potential_chunk) <= self.max_chunk_size:
                        current_chunk = potential_chunk
                    else:
                        if current_chunk:
                            result.append(current_chunk)
                        current_chunk = part
                
                if current_chunk:
                    result.append(current_chunk)
                
                # กรองที่เล็กเกินไป
                result = [chunk for chunk in result if len(chunk) >= self.min_chunk_size]
                
                if result:
                    return result
        
        # ถ้าแบ่งไม่ได้ ใช้วิธีสุดท้าย
        return self._fallback_split(content)
    
    def _fallback_split(self, text: str) -> List[str]:
        """วิธีสำรองเมื่อแบ่งตามโครงสร้างไม่ได้"""
        separators = ['\n\n', '\n', '. ', ' ', '']
        return self._recursive_split(text, separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting ที่ปรับปรุงแล้ว"""
        
        if len(text) <= self.max_chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []
        
        if not separators:
            # ถ้าไม่มี separator แล้ว ตัดแบบ hard
            return [text[i:i+self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size)]
        
        separator = separators[0]
        if separator:
            splits = text.split(separator)
        else:
            splits = [text[i:i+self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size)]
        
        good_splits = []
        for split in splits:
            if len(split) <= self.max_chunk_size:
                if len(split) >= self.min_chunk_size:
                    good_splits.append(split)
            else:
                sub_splits = self._recursive_split(split, separators[1:])
                good_splits.extend(sub_splits)
        
        return self._merge_splits(good_splits, separator)
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """รวมชิ้นเล็กเป็นชิ้นใหญ่อย่างชาญฉลาด"""
        
        if not splits:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split = split.strip()
            if not split:
                continue
                
            split_length = len(split)
            separator_length = len(separator) if current_chunk else 0
            potential_length = current_length + split_length + separator_length
            
            if potential_length <= self.max_chunk_size:
                current_chunk.append(split)
                current_length = potential_length
            else:
                if current_chunk:
                    merged = separator.join(current_chunk)
                    if len(merged) >= self.min_chunk_size:
                        chunks.append(merged)
                current_chunk = [split]
                current_length = split_length
        
        if current_chunk:
            merged = separator.join(current_chunk)
            if len(merged) >= self.min_chunk_size:
                chunks.append(merged)
        
        return chunks
    
    def _handle_overlaps_and_merge(self, chunks: List[str]) -> List[str]:
        """จัดการ overlap และรวม chunks เล็ก"""
        if not chunks:
            return []
        
        # เพิ่ม overlap ระหว่าง chunks
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # เพิ่ม overlap จาก chunk ก่อนหน้า
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-self.overlap_size:] if len(prev_chunk) > self.overlap_size else prev_chunk
                
                # หา boundary ที่เหมาะสม (จุดจบประโยคหรือย่อหน้า)
                overlap_text = self._find_good_overlap_boundary(overlap_text)
                
                combined_chunk = overlap_text + '\n' + chunk
                if len(combined_chunk) <= self.max_chunk_size * 1.2:  # อนุญาตให้เกินขนาดเล็กน้อย
                    overlapped_chunks.append(combined_chunk)
                else:
                    overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    def _find_good_overlap_boundary(self, text: str) -> str:
        """หาจุดตัดที่เหมาะสมสำหรับ overlap"""
        if len(text) <= self.overlap_size:
            return text
        
        # หาจุดจบประโยค
        sentence_end = -1
        for i in range(len(text)-1, max(0, len(text) - self.overlap_size - 50), -1):
            if text[i] in '.!?。':
                sentence_end = i + 1
                break
        
        if sentence_end > 0:
            return text[sentence_end:].strip()
        
        # หาจุดจบคำ
        word_boundary = text.rfind(' ', max(0, len(text) - self.overlap_size))
        if word_boundary > len(text) - self.overlap_size - 20:
            return text[word_boundary:].strip()
        
        # ใช้ขนาดเดิม
        return text[-self.overlap_size:]
    
    def _add_enhanced_metadata(self, chunks: List[str], structure: Dict, source_file: str = "") -> List[Dict]:
        """เพิ่ม metadata ที่ปรับปรุงแล้ว"""
        
        enriched_chunks = []
        
        for i, chunk in enumerate(chunks):
            # คำนวณ hash สำหรับ deduplication
            chunk_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()[:12]
            
            # วิเคราะห์เนื้อหา chunk
            chunk_analysis = self._analyze_chunk_content(chunk)
            
            # ตรวจสอบ header hierarchy ใน chunk
            chunk_headers = self._identify_chunk_headers(chunk)
            
            metadata = {
                'chunk_id': i,
                'chunk_hash': chunk_hash,
                'content': chunk,
                'length': len(chunk),
                'word_count': len(chunk.split()),
                'source_file': source_file,
                'document_title': structure.get('title', ''),
                'document_type': structure.get('document_type', 'general'),
                'section_type': self._identify_section_type(chunk),
                'header_info': chunk_headers,
                'contains_table': self._contains_table(chunk),
                'contains_course_codes': bool(self._extract_course_codes(chunk)),
                'main_topics': self._extract_topics(chunk),
                'course_codes': list(self._extract_course_codes(chunk)),
                'quality_score': self._calculate_quality_score(chunk, chunk_analysis),
                'readability_score': chunk_analysis['readability'],
                'information_density': chunk_analysis['info_density'],
                'structure_score': chunk_analysis['structure'],
                'language_quality': chunk_analysis['language'],
                'created_at': datetime.now().isoformat(),
                'processing_version': '2.1'
            }
            
            enriched_chunks.append(metadata)
        
        return enriched_chunks
    
    def _identify_chunk_headers(self, chunk: str) -> Dict:
        """ระบุ header hierarchy ใน chunk"""
        headers = {
            'main_headers': [],
            'sub_headers': [],
            'sub_sub_headers': [],
            'has_hierarchy': False
        }
        
        lines = chunk.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^#\s+(.+)$', line):
                headers['main_headers'].append(line)
            elif re.match(r'^##\s+(.+)$', line):
                headers['sub_headers'].append(line)
            elif re.match(r'^###\s+(.+)$', line):
                headers['sub_sub_headers'].append(line)
        
        headers['has_hierarchy'] = any([
            headers['main_headers'],
            headers['sub_headers'],
            headers['sub_sub_headers']
        ])
        
        return headers
    
    def _analyze_chunk_content(self, chunk: str) -> Dict:
        """วิเคราะห์คุณภาพและลักษณะของ chunk"""
        
        analysis = {
            'readability': 0.0,
            'info_density': 0.0,
            'structure': 0.0,
            'language': 0.0
        }
        
        if not chunk.strip():
            return analysis
        
        # คำนวณ readability (ความยาวประโยคเฉลี่ย)
        sentences = re.split(r'[.!?。]', chunk)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            analysis['readability'] = max(0, min(1, 1 - (avg_sentence_length - 10) / 20))  # Optimal ~10 words
        
        # คำนวณ information density
        words = chunk.split()
        if words:
            unique_words = set(words)
            analysis['info_density'] = len(unique_words) / len(words)
        
        # คำนวณ structure score
        structure_indicators = 0
        structure_indicators += len(re.findall(r'[๑-๙]+\.', chunk))  # numbered lists
        structure_indicators += len(re.findall(r'##+', chunk))  # headers
        structure_indicators += len(re.findall(r'\n\n', chunk))  # paragraphs
        
        analysis['structure'] = min(1.0, structure_indicators / 5)  # normalize to 0-1
        
        # คำนวณ language quality (อัตราส่วนตัวอักษรไทยต่อทั้งหมด)
        thai_chars = len(re.findall(r'[ก-๙]', chunk))
        total_chars = len(re.sub(r'\s', '', chunk))
        analysis['language'] = thai_chars / total_chars if total_chars > 0 else 0
        
        return analysis
    
    def _calculate_quality_score(self, chunk: str, analysis: Dict) -> float:
        """คำนวณคะแนนคุณภาพโดยรวม"""
        
        weights = {
            'readability': 0.25,
            'info_density': 0.25,
            'structure': 0.25,
            'language': 0.25
        }
        
        score = sum(analysis[key] * weights[key] for key in weights)
        
        # ปรับแต่งตามความยาว
        length = len(chunk)
        if length < self.min_chunk_size:
            score *= 0.5
        elif length > self.max_chunk_size * 1.5:
            score *= 0.7
        
        # โบนัสสำหรับเนื้อหาที่มีโครงสร้าง
        if self._contains_table(chunk):
            score += 0.1
        if self._extract_course_codes(chunk):
            score += 0.1
        
        # โบนัสสำหรับ header hierarchy
        headers = self._identify_chunk_headers(chunk)
        if headers['has_hierarchy']:
            score += 0.05
            
        return min(1.0, max(0.0, score))
    
    def _identify_section_type(self, chunk: str) -> str:
        """ระบุประเภทของส่วนด้วยอัลกอริทึมที่ปรับปรุง"""
        
        type_scores = {}
        chunk_lower = chunk.lower()
        
        for doc_type, keywords in self.document_types.items():
            score = sum(chunk_lower.count(keyword.lower()) for keyword in keywords)
            if score > 0:
                type_scores[doc_type] = score
        
        return max(type_scores, key=type_scores.get) if type_scores else 'general'
    
    def _contains_table(self, chunk: str) -> bool:
        """ตรวจสอบว่ามีตารางหรือไม่"""
        for pattern in self.table_patterns:
            if re.search(pattern, chunk, re.DOTALL):
                return True
        return False
    
    def _extract_course_codes(self, chunk: str) -> set:
        """ดึงรหัสวิชาทั้งหมด"""
        codes = set()
        for pattern in self.course_code_patterns:
            matches = re.findall(pattern, chunk)
            codes.update(matches)
        return codes
    
    def _extract_topics(self, chunk: str) -> List[str]:
        """ดึงหัวข้อสำคัญด้วยวิธีที่ปรับปรุง"""
        
        topics = []
        
        # รหัสวิชา
        topics.extend(list(self._extract_course_codes(chunk)))
        
        # ชื่อหลักสูตร
        curriculum_patterns = [
            r'หลักสูตร([^(]+?)(?:\(|$)',
            r'สาขาวิชา([^(]+?)(?:\(|$)',
            r'สาขา([^(]+?)(?:\(|$)'
        ]
        
        for pattern in curriculum_patterns:
            matches = re.findall(pattern, chunk)
            topics.extend([match.strip() for match in matches])
        
        # ชื่อวิชา
        subject_patterns = [
            r'วิชา([ก-๙\s]+?)(?:\s+[A-Z]|\n|$)',
            r'รายวิชา([ก-๙\s]+?)(?:\s+[A-Z]|\n|$)',
        ]
        
        for pattern in subject_patterns:
            matches = re.findall(pattern, chunk)
            topics.extend([match.strip() for match in matches if len(match.strip()) > 3])
        
        # คณะและภาควิชา
        org_patterns = [
            r'คณะ([ก-๙\s]+?)(?:\s|$|\.)',
            r'ภาควิชา([ก-๙\s]+?)(?:\s|$|\.)',
            r'สำนัก([ก-๙\s]+?)(?:\s|$|\.)'
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, chunk)
            topics.extend([match.strip() for match in matches if len(match.strip()) > 3])
        
        # ลบของซ้ำและกรองคำสั้น
        unique_topics = list(set([topic for topic in topics if len(topic) > 2]))
        return unique_topics[:10]  # จำกัดไม่เกิน 10 topics
    
    def save_enhanced_results(self, results: Dict[str, List[Dict]], output_path: str, include_analytics: bool = True):
        """บันทึกผลลัพธ์พร้อมการวิเคราะห์"""
        
        # สร้าง output directory ถ้าไม่มี
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # คำนวณสถิติ
        stats = self._calculate_statistics(results)
        
        output_data = {
            'metadata': {
                'processing_date': datetime.now().isoformat(),
                'version': '2.1',
                'total_files': len(results),
                'total_chunks': sum(len(chunks) for chunks in results.values()),
                'processing_settings': {
                    'max_chunk_size': self.max_chunk_size,
                    'overlap_size': self.overlap_size,
                    'min_chunk_size': self.min_chunk_size,
                    'quality_threshold': self.quality_threshold
                }
            },
            'chunks': results
        }
        
        if include_analytics:
            output_data['analytics'] = stats
        
        # บันทึกไฟล์หลัก
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # บันทึกไฟล์สถิติแยก
        stats_path = output_path.with_suffix('').with_suffix('_analytics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # บันทึก summary report
        report_path = output_path.with_suffix('.txt')
        self._generate_report(results, stats, report_path)
        
        self.logger.info(f"Results saved to {output_path}")
        self.logger.info(f"Analytics saved to {stats_path}")
        self.logger.info(f"Report saved to {report_path}")
        
        return stats
    
    def _calculate_statistics(self, results: Dict[str, List[Dict]]) -> Dict:
        """คำนวณสถิติโดยละเอียด"""
        
        all_chunks = [chunk for chunks in results.values() for chunk in chunks]
        
        if not all_chunks:
            return {}
        
        # สถิติพื้นฐาน
        stats = {
            'basic_stats': {
                'total_files': len(results),
                'total_chunks': len(all_chunks),
                'avg_chunks_per_file': len(all_chunks) / len(results) if results else 0,
                'total_content_length': sum(chunk['length'] for chunk in all_chunks),
                'total_word_count': sum(chunk['word_count'] for chunk in all_chunks)
            },
            
            'chunk_size_distribution': {
                'min_size': min(chunk['length'] for chunk in all_chunks),
                'max_size': max(chunk['length'] for chunk in all_chunks),
                'avg_size': sum(chunk['length'] for chunk in all_chunks) / len(all_chunks),
                'median_size': sorted([chunk['length'] for chunk in all_chunks])[len(all_chunks)//2]
            },
            
            'quality_scores': {
                'avg_quality': sum(chunk['quality_score'] for chunk in all_chunks) / len(all_chunks),
                'avg_readability': sum(chunk['readability_score'] for chunk in all_chunks) / len(all_chunks),
                'avg_info_density': sum(chunk['information_density'] for chunk in all_chunks) / len(all_chunks),
                'high_quality_chunks': len([c for c in all_chunks if c['quality_score'] > 0.8]),
                'low_quality_chunks': len([c for c in all_chunks if c['quality_score'] < 0.5])
            },
            
            'content_analysis': {
                'document_types': self._count_by_field(all_chunks, 'document_type'),
                'section_types': self._count_by_field(all_chunks, 'section_type'),
                'files_with_tables': len([f for f, chunks in results.items() 
                                        if any(c['contains_table'] for c in chunks)]),
                'files_with_course_codes': len([f for f, chunks in results.items() 
                                              if any(c['contains_course_codes'] for c in chunks)]),
                'chunks_with_headers': len([c for c in all_chunks 
                                          if c.get('header_info', {}).get('has_hierarchy', False)])
            },
            
            'header_analysis': {
                'main_headers_count': sum(len(c.get('header_info', {}).get('main_headers', [])) for c in all_chunks),
                'sub_headers_count': sum(len(c.get('header_info', {}).get('sub_headers', [])) for c in all_chunks),
                'sub_sub_headers_count': sum(len(c.get('header_info', {}).get('sub_sub_headers', [])) for c in all_chunks)
            },
            
            'top_topics': self._get_top_topics(all_chunks),
            'top_course_codes': self._get_top_course_codes(all_chunks),
        }
        
        return stats
    
    def _count_by_field(self, chunks: List[Dict], field: str) -> Dict:
        """นับจำนวนตาม field ที่กำหนด"""
        counts = {}
        for chunk in chunks:
            value = chunk.get(field, 'unknown')
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    def _get_top_topics(self, chunks: List[Dict], top_n: int = 20) -> List[Tuple[str, int]]:
        """หาหัวข้อที่พบบ่อยที่สุด"""
        topic_counts = {}
        for chunk in chunks:
            for topic in chunk.get('main_topics', []):
                if len(topic) > 3:  # กรองคำสั้น
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def _get_top_course_codes(self, chunks: List[Dict], top_n: int = 50) -> List[Tuple[str, int]]:
        """หารหัสวิชาที่พบบ่อยที่สุด"""
        code_counts = {}
        for chunk in chunks:
            for code in chunk.get('course_codes', []):
                code_counts[code] = code_counts.get(code, 0) + 1
        
        return sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def _generate_report(self, results: Dict[str, List[Dict]], stats: Dict, report_path: Path):
        """สร้างรายงานสรุปผล"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED DOCUMENT PROCESSING REPORT v2.1\n")
            f.write("=" * 60 + "\n\n")
            
            # สถิติพื้นฐาน
            basic = stats['basic_stats']
            f.write(f"BASIC STATISTICS:\n")
            f.write(f"   Total Files: {basic['total_files']:,}\n")
            f.write(f"   Total Chunks: {basic['total_chunks']:,}\n")
            f.write(f"   Avg Chunks/File: {basic['avg_chunks_per_file']:.1f}\n")
            f.write(f"   Total Content: {basic['total_content_length']:,} characters\n")
            f.write(f"   Total Words: {basic['total_word_count']:,}\n\n")
            
            # การกระจายขนาด chunk
            size_dist = stats['chunk_size_distribution']
            f.write(f"CHUNK SIZE DISTRIBUTION:\n")
            f.write(f"   Min Size: {size_dist['min_size']:,}\n")
            f.write(f"   Max Size: {size_dist['max_size']:,}\n")
            f.write(f"   Average: {size_dist['avg_size']:.0f}\n")
            f.write(f"   Median: {size_dist['median_size']:,}\n\n")
            
            # คะแนนคุณภาพ
            quality = stats['quality_scores']
            f.write(f"QUALITY METRICS:\n")
            f.write(f"   Avg Quality Score: {quality['avg_quality']:.3f}\n")
            f.write(f"   Avg Readability: {quality['avg_readability']:.3f}\n")
            f.write(f"   Avg Info Density: {quality['avg_info_density']:.3f}\n")
            f.write(f"   High Quality Chunks: {quality['high_quality_chunks']}\n")
            f.write(f"   Low Quality Chunks: {quality['low_quality_chunks']}\n\n")
            
            # การวิเคราะห์ header
            header_stats = stats['header_analysis']
            f.write(f"HEADER HIERARCHY ANALYSIS:\n")
            f.write(f"   Main Headers (#): {header_stats['main_headers_count']}\n")
            f.write(f"   Sub Headers (##): {header_stats['sub_headers_count']}\n")
            f.write(f"   Sub-Sub Headers (###): {header_stats['sub_sub_headers_count']}\n")
            f.write(f"   Chunks with Headers: {stats['content_analysis']['chunks_with_headers']}\n\n")
            
            # ประเภทเอกสาร
            f.write(f"DOCUMENT TYPES:\n")
            for doc_type, count in stats['content_analysis']['document_types'].items():
                f.write(f"   {doc_type}: {count}\n")
            f.write("\n")
            
            # หัวข้อยอดนิยม
            f.write(f"TOP 10 TOPICS:\n")
            for topic, count in stats['top_topics'][:10]:
                f.write(f"   {topic}: {count}\n")
            f.write("\n")
            
            # รหัสวิชายอดนิยม
            f.write(f"TOP 10 COURSE CODES:\n")
            for code, count in stats['top_course_codes'][:10]:
                f.write(f"   {code}: {count}\n")
            f.write("\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def process_year_2568_by_meetings():
    """ประมวลผลแบบแยกตามการประชุมแต่ละครั้ง"""
    
    print("Starting Enhanced RAG Document Processing v2.1 - Organized by Meetings...")
    print("With complete header hierarchy support (# ## ###)")
    
    # สร้าง enhanced splitter
    splitter = AcademicDocumentSplitter(
        max_chunk_size=1200,
        overlap_size=150,
        min_chunk_size=200,
        preserve_sections=True,
        preserve_tables=True,
        quality_threshold=0.3
    )
    
    base_folder = Path("ocr_output/ประจำปี 2568")
    
    if not base_folder.exists():
        print(f"Folder not found: {base_folder}")
        return None
    
    # หาโฟลเดอร์การประชุมทั้งหมด
    meeting_folders = [f for f in base_folder.iterdir() if f.is_dir()]
    
    if not meeting_folders:
        print("No meeting folders found!")
        return None
    
    print(f"Found {len(meeting_folders)} meetings to process")
    
    # สร้างโฟลเดอร์ output
    output_base = Path("processed_chunks_2568_v2")
    output_base.mkdir(exist_ok=True)
    
    all_results = {}
    summary_stats = {
        'meetings': {},
        'totals': {
            'total_meetings': 0,
            'total_files': 0,
            'total_chunks': 0,
            'processing_time': 0
        }
    }
    
    start_time = datetime.now()
    
    for i, meeting_folder in enumerate(sorted(meeting_folders), 1):
        meeting_name = meeting_folder.name
        print(f"\n[{i}/{len(meeting_folders)}] Processing: {meeting_name}")
        
        try:
            # ประมวลผลแต่ละการประชุม
            meeting_start = datetime.now()
            meeting_results = splitter.process_files_recursive(meeting_folder)
            meeting_end = datetime.now()
            
            if not meeting_results:
                print(f"   No files processed in {meeting_name}")
                continue
            
            # สรุปสถิติการประชุม
            meeting_chunks = sum(len(chunks) for chunks in meeting_results.values())
            meeting_time = (meeting_end - meeting_start).total_seconds()
            
            print(f"   Files: {len(meeting_results)}")
            print(f"   Chunks: {meeting_chunks}")
            print(f"   Time: {meeting_time:.1f}s")
            
            # บันทึกไฟล์แยกตามการประชุม
            safe_meeting_name = re.sub(r'[^\w\s-]', '', meeting_name).strip()
            safe_meeting_name = re.sub(r'[-\s]+', '_', safe_meeting_name)
            
            meeting_output_file = output_base / f"meeting_{safe_meeting_name}.json"
            meeting_stats = splitter.save_enhanced_results(
                meeting_results, 
                str(meeting_output_file),
                include_analytics=True
            )
            
            # เก็บสถิติ
            summary_stats['meetings'][meeting_name] = {
                'files': len(meeting_results),
                'chunks': meeting_chunks,
                'processing_time': meeting_time,
                'output_file': str(meeting_output_file),
                'avg_quality': meeting_stats['quality_scores']['avg_quality'] if meeting_stats else 0,
                'document_types': meeting_stats['content_analysis']['document_types'] if meeting_stats else {},
                'header_counts': meeting_stats.get('header_analysis', {}) if meeting_stats else {}
            }
            
            all_results[meeting_name] = meeting_results
            
        except Exception as e:
            print(f"   Error processing {meeting_name}: {e}")
            splitter.logger.error(f"Failed to process {meeting_name}: {e}")
            continue
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # สรุปสถิติรวม
    summary_stats['totals'] = {
        'total_meetings': len(summary_stats['meetings']),
        'total_files': sum(stats['files'] for stats in summary_stats['meetings'].values()),
        'total_chunks': sum(stats['chunks'] for stats in summary_stats['meetings'].values()),
        'processing_time': total_time,
        'avg_files_per_meeting': sum(stats['files'] for stats in summary_stats['meetings'].values()) / len(summary_stats['meetings']) if summary_stats['meetings'] else 0,
        'processing_speed_chunks_per_sec': sum(stats['chunks'] for stats in summary_stats['meetings'].values()) / total_time if total_time > 0 else 0
    }
    
    # บันทึกไฟล์สรุปรวม
    summary_file = output_base / "summary_all_meetings_v2.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)
    
    # สร้างรายงานสรุป
    create_summary_report(summary_stats, output_base / "summary_report_v2.txt")
    
    # แสดงผลสรุปรวม
    print(f"\nALL MEETINGS PROCESSING COMPLETED!")
    print("=" * 60)
    print(f"Version: 2.1 (with complete header hierarchy support)")
    print(f"Meetings processed: {summary_stats['totals']['total_meetings']}")
    print(f"Total files: {summary_stats['totals']['total_files']:,}")
    print(f"Total chunks: {summary_stats['totals']['total_chunks']:,}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Speed: {summary_stats['totals']['processing_speed_chunks_per_sec']:.1f} chunks/sec")
    print(f"Output folder: {output_base.absolute()}")
    
    print(f"\nTOP 5 MEETINGS BY CHUNKS:")
    top_meetings = sorted(
        summary_stats['meetings'].items(), 
        key=lambda x: x[1]['chunks'], 
        reverse=True
    )[:5]
    
    for meeting_name, stats in top_meetings:
        print(f"   {meeting_name[:50]}: {stats['chunks']} chunks")
    
    return all_results, summary_stats


def create_summary_report(summary_stats: Dict, report_path: Path):
    """สร้างรายงานสรุปที่อ่านง่าย"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAG DOCUMENT PROCESSING SUMMARY REPORT v2.1\n")
        f.write("=" * 60 + "\n")
        f.write("Enhanced with complete header hierarchy support (# ## ###)\n\n")
        
        # สถิติรวม
        totals = summary_stats['totals']
        f.write("OVERALL STATISTICS:\n")
        f.write(f"   Total meetings processed: {totals['total_meetings']}\n")
        f.write(f"   Total files processed: {totals['total_files']:,}\n")
        f.write(f"   Total chunks generated: {totals['total_chunks']:,}\n")
        f.write(f"   Average files per meeting: {totals['avg_files_per_meeting']:.1f}\n")
        f.write(f"   Processing time: {totals['processing_time']:.1f} seconds\n")
        f.write(f"   Processing speed: {totals['processing_speed_chunks_per_sec']:.1f} chunks/second\n\n")
        
        # รายละเอียดแต่ละการประชุม
        f.write("MEETINGS BREAKDOWN:\n")
        f.write("-" * 60 + "\n")
        
        meetings = sorted(summary_stats['meetings'].items(), key=lambda x: x[1]['chunks'], reverse=True)
        
        for meeting_name, stats in meetings:
            f.write(f"\n{meeting_name}\n")
            f.write(f"   Files: {stats['files']}\n")
            f.write(f"   Chunks: {stats['chunks']}\n")
            f.write(f"   Avg Quality: {stats['avg_quality']:.3f}\n")
            f.write(f"   Processing Time: {stats['processing_time']:.1f}s\n")
            f.write(f"   Output: {Path(stats['output_file']).name}\n")
            
            # แสดงประเภทเอกสาร
            if stats['document_types']:
                f.write(f"   Document Types: ")
                types_str = ", ".join([f"{k}({v})" for k, v in stats['document_types'].items()])
                f.write(f"{types_str}\n")
            
            # แสดง header counts ถ้ามี
            if stats.get('header_counts'):
                header_counts = stats['header_counts']
                if any(header_counts.values()):
                    f.write(f"   Headers: ")
                    header_info = []
                    if header_counts.get('main_headers_count', 0) > 0:
                        header_info.append(f"Main({header_counts['main_headers_count']})")
                    if header_counts.get('sub_headers_count', 0) > 0:
                        header_info.append(f"Sub({header_counts['sub_headers_count']})")
                    if header_counts.get('sub_sub_headers_count', 0) > 0:
                        header_info.append(f"Sub-Sub({header_counts['sub_sub_headers_count']})")
                    f.write(", ".join(header_info) + "\n")
        
        f.write(f"\n\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def quick_browse_meetings():
    """ดูรายการการประชุมที่มี"""
    
    base_folder = Path("ocr_output/ประจำปี 2568")
    
    if not base_folder.exists():
        print(f"Folder not found: {base_folder}")
        return
    
    meeting_folders = [f for f in base_folder.iterdir() if f.is_dir()]
    
    if not meeting_folders:
        print("No meeting folders found!")
        return
    
    print(f"Found {len(meeting_folders)} meetings:")
    print("=" * 60)
    
    for i, meeting_folder in enumerate(sorted(meeting_folders), 1):
        txt_files = list(meeting_folder.glob("*.txt"))
        print(f"{i:2d}. {meeting_folder.name}")
        print(f"    Files: {len(txt_files)}")
        
        if txt_files:
            total_size = sum(f.stat().st_size for f in txt_files)
            print(f"    Total size: {total_size/1024:.1f} KB")
        print()


def quick_analyze_file(file_path: str, max_chunk_size: int = 1000):
    """วิเคราะห์ไฟล์เดียวอย่างรวดเร็ว"""
    
    splitter = AcademicDocumentSplitter(
        max_chunk_size=max_chunk_size,
        quality_threshold=0.2
    )
    
    content = splitter.read_file(file_path)
    if not content:
        print(f"Cannot read file: {file_path}")
        return
    
    chunks = splitter.split_document(content, source_file=file_path)
    
    print(f"\nFILE ANALYSIS: {Path(file_path).name}")
    print("=" * 50)
    print(f"File size: {len(content):,} characters")
    print(f"Generated chunks: {len(chunks)}")
    
    if chunks:
        avg_quality = sum(c['quality_score'] for c in chunks) / len(chunks)
        print(f"Average quality: {avg_quality:.3f}")
        print(f"Document type: {chunks[0]['document_type']}")
        print(f"Document title: {chunks[0]['document_title'][:60]}...")
        
        # Header analysis
        total_headers = sum(
            len(c.get('header_info', {}).get('main_headers', [])) +
            len(c.get('header_info', {}).get('sub_headers', [])) +
            len(c.get('header_info', {}).get('sub_sub_headers', []))
            for c in chunks
        )
        print(f"Total headers found: {total_headers}")
        
        print(f"\nCHUNK BREAKDOWN:")
        for i, chunk in enumerate(chunks[:5]):  # แสดง 5 chunks แรก
            headers_info = chunk.get('header_info', {})
            header_count = (len(headers_info.get('main_headers', [])) + 
                           len(headers_info.get('sub_headers', [])) + 
                           len(headers_info.get('sub_sub_headers', [])))
            
            print(f"  [{i+1}] {chunk['length']} chars, Q:{chunk['quality_score']:.3f}, "
                  f"Type:{chunk['section_type']}, Headers:{header_count}")
        
        if len(chunks) > 5:
            print(f"  ... และอีก {len(chunks)-5} chunks")
    
    return chunks


if __name__ == "__main__":
    # เลือกโหมดการทำงาน
    print("Enhanced RAG Document Processing v2.1 - Choose Mode:")
    print("1. Process all meetings separately with header hierarchy support (Recommended)")
    print("2. Browse available meetings")
    print("3. Quick analyze single file")
    
    # สำหรับการรันอัตโนมัติ ให้ใช้โหมด 1
    mode = "1"
    
    if mode == "1":
        results, stats = process_year_2568_by_meetings()
    elif mode == "2":
        quick_browse_meetings()
    elif mode == "3":
        # ตัวอย่างการใช้ quick_analyze_file
        # quick_analyze_file("path/to/your/file.txt", max_chunk_size=1000)
        pass
    else:
        print("Invalid mode selected")