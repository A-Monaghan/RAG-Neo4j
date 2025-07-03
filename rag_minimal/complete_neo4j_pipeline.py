#!/usr/bin/env python3
"""
Complete Neo4j RAG Pipeline

This script orchestrates the complete workflow:
1. Enhanced PDF processing with PyMuPDF
2. Information extraction (NER + relationships)
3. Embedding generation
4. Neo4j ingestion with batching
5. Graph-based RAG functionality

Designed to process all your PDFs into a Neo4j knowledge graph.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

# Import our custom modules
from enhanced_pdf_processor import (
    process_pdf_for_neo4j, 
    batch_process_pdfs,
    DocumentMetadata,
    TextChunk
)

from neo4j_information_extraction import (
    InformationExtractor,
    process_document_chunks
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neo4j_rag_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Neo4jRAGPipeline:
    """Complete RAG pipeline for Neo4j ingestion."""
    
    def __init__(self, claude_api_key: Optional[str] = None,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password"):
        """Initialize the complete pipeline."""
        
        self.claude_api_key = claude_api_key or os.getenv('ANTHROPIC_API_KEY')
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        # Initialize components
        self.information_extractor = None
        self.embedding_model = None
        self.neo4j_db = None
        
        logger.info("🚀 Neo4j RAG Pipeline initialized")
    
    def setup_components(self):
        """Setup all pipeline components."""
        
        logger.info("🔧 Setting up pipeline components...")
        
        try:
            # 1. Information Extraction
            from neo4j_information_extraction import InformationExtractor
            self.information_extractor = InformationExtractor(self.claude_api_key)
            logger.info("✅ Information extraction setup complete")
            
            # 2. Embedding Model
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded")
            
            # 3. Neo4j Database (optional - may not be running)
            try:
                from neo4j_database import Neo4jRAGDatabase
                self.neo4j_db = Neo4jRAGDatabase(
                    uri=self.neo4j_uri,
                    username=self.neo4j_user,
                    password=self.neo4j_password
                )
                self.neo4j_db.setup_schema()
                logger.info("✅ Neo4j database connected")
            except Exception as e:
                logger.warning(f"⚠️ Neo4j not available: {e}")
                logger.info("💡 Will save data to JSON files instead")
                self.neo4j_db = None
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Component setup failed: {e}")
            return False
    
    def process_single_document(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF through the complete pipeline."""
        
        logger.info(f"📄 Processing: {Path(pdf_path).name}")
        
        try:
            # Step 1: PDF Processing
            document_data = process_pdf_for_neo4j(pdf_path)
            if not document_data:
                logger.warning(f"❌ Failed to process PDF: {pdf_path}")
                return None
            
            # Step 2: Information Extraction
            chunks = document_data['chunks']
            extraction_results = []
            
            if self.information_extractor:
                logger.info(f"🧠 Extracting information from {len(chunks)} chunks...")
                extraction_results = process_document_chunks(chunks, self.claude_api_key)
            else:
                logger.warning("⚠️ Information extraction not available")
                extraction_results = [{'entities': [], 'relationships': [], 'concepts': []} for _ in chunks]
            
            # Step 3: Generate Embeddings
            embeddings = []
            if self.embedding_model:
                logger.info("🔢 Generating embeddings...")
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = self.embedding_model.encode(chunk_texts).tolist()
                logger.info(f"✅ Generated {len(embeddings)} embeddings")
            
            # Step 4: Prepare complete result
            result = {
                'document_data': document_data,
                'extraction_results': extraction_results,
                'embeddings': embeddings,
                'processing_timestamp': datetime.now().isoformat(),
                'stats': {
                    'chunks': len(chunks),
                    'entities': sum(len(er.get('entities', [])) for er in extraction_results),
                    'relationships': sum(len(er.get('relationships', [])) for er in extraction_results),
                    'concepts': sum(len(er.get('concepts', [])) for er in extraction_results)
                }
            }
            
            logger.info(f"✅ Document processed: {result['stats']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def ingest_to_neo4j(self, processed_document: Dict[str, Any]) -> bool:
        """Ingest processed document to Neo4j."""
        
        if not self.neo4j_db:
            logger.warning("⚠️ Neo4j not available - skipping ingestion")
            return False
        
        try:
            document_data = processed_document['document_data']
            extraction_results = processed_document['extraction_results']
            embeddings = processed_document['embeddings']
            
            # Use Neo4j batch ingestion
            stats = self.neo4j_db.batch_ingest_document(
                document_data=document_data,
                extraction_data=extraction_results,
                embeddings=embeddings
            )
            
            logger.info(f"✅ Neo4j ingestion complete: {stats}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Neo4j ingestion failed: {e}")
            return False
    
    def save_to_json(self, processed_document: Dict[str, Any], output_dir: str = "neo4j_output"):
        """Save processed document to JSON files."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Get document filename
            doc_filename = processed_document['document_data']['document'].filename
            base_name = Path(doc_filename).stem
            
            # Save complete data
            json_file = output_path / f"{base_name}_processed.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(processed_document, f, indent=2, default=str)
            
            logger.info(f"💾 Saved to: {json_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving JSON: {e}")
            return False
    
    def process_pdf_directory(self, pdf_directory: str = "pdfs", 
                            max_files: Optional[int] = None,
                            save_json: bool = True) -> Dict[str, int]:
        """Process all PDFs in directory."""
        
        logger.info(f"📁 Processing PDFs from: {pdf_directory}")
        
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            logger.error(f"❌ Directory not found: {pdf_directory}")
            return {'processed': 0, 'failed': 0, 'ingested': 0}
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        logger.info(f"📋 Found {len(pdf_files)} PDF files to process")
        
        stats = {'processed': 0, 'failed': 0, 'ingested': 0}
        
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"\n🔄 Processing {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                # Process document
                processed_doc = self.process_single_document(str(pdf_file))
                
                if processed_doc:
                    stats['processed'] += 1
                    
                    # Save to JSON
                    if save_json:
                        self.save_to_json(processed_doc)
                    
                    # Ingest to Neo4j
                    if self.ingest_to_neo4j(processed_doc):
                        stats['ingested'] += 1
                        
                else:
                    stats['failed'] += 1
                    logger.warning(f"⚠️ Failed to process: {pdf_file.name}")
                
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"❌ Error with {pdf_file.name}: {e}")
        
        logger.info(f"\n📊 PROCESSING COMPLETE: {stats}")
        return stats
    
    def create_rag_demo(self, output_dir: str = "neo4j_output"):
        """Create a RAG demonstration script."""
        
        rag_demo_code = '''#!/usr/bin/env python3
"""
Neo4j RAG Demonstration

Query your processed documents using graph-based retrieval.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

class Neo4jRAGDemo:
    """Demonstration of Neo4j-based RAG capabilities."""
    
    def __init__(self, data_directory: str = "neo4j_output"):
        self.data_dir = Path(data_directory)
        self.documents = self.load_processed_documents()
        print(f"📚 Loaded {len(self.documents)} processed documents")
    
    def load_processed_documents(self) -> List[Dict[str, Any]]:
        """Load all processed JSON documents."""
        documents = []
        
        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return documents
        
        for json_file in self.data_dir.glob("*_processed.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                    documents.append(doc_data)
            except Exception as e:
                print(f"⚠️ Error loading {json_file}: {e}")
        
        return documents
    
    def search_documents(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Simple text-based search through documents."""
        results = []
        query_lower = query.lower()
        
        for doc in self.documents:
            chunks = doc['document_data']['chunks']
            
            for chunk in chunks:
                if query_lower in chunk['text'].lower():
                    results.append({
                        'document': doc['document_data']['document']['filename'],
                        'chunk_id': chunk['chunk_id'],
                        'text': chunk['text'][:200] + "...",
                        'word_count': chunk['word_count']
                    })
                    
                    if len(results) >= max_results:
                        return results
        
        return results
    
    def get_entities_for_query(self, query: str) -> List[Dict[str, Any]]:
        """Find entities related to query."""
        entities = []
        query_lower = query.lower()
        
        for doc in self.documents:
            for extraction in doc['extraction_results']:
                for entity in extraction.get('entities', []):
                    if query_lower in entity['text'].lower():
                        entities.append({
                            'entity': entity['text'],
                            'type': entity['label'],
                            'document': doc['document_data']['document']['filename']
                        })
        
        return entities
    
    def demo_queries(self):
        """Run demonstration queries."""
        
        print("\\n🔍 RAG DEMONSTRATION")
        print("=" * 50)
        
        sample_queries = [
            "environmental protection",
            "development proposal", 
            "legal case",
            "government policy"
        ]
        
        for query in sample_queries:
            print(f"\\n🎯 Query: '{query}'")
            print("-" * 30)
            
            # Search results
            results = self.search_documents(query, max_results=3)
            print(f"📄 Found {len(results)} relevant chunks")
            
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['document']}")
                print(f"      {result['text']}")
                print()
            
            # Related entities
            entities = self.get_entities_for_query(query)
            if entities:
                print(f"🏷️ Related entities: {len(entities)}")
                for entity in entities[:3]:
                    print(f"   • {entity['entity']} ({entity['type']})")
            
            print()

if __name__ == "__main__":
    demo = Neo4jRAGDemo()
    demo.demo_queries()
'''
        
        # Save RAG demo script
        demo_file = Path(output_dir) / "rag_demo.py"
        with open(demo_file, 'w', encoding='utf-8') as f:
            f.write(rag_demo_code)
        
        logger.info(f"🎯 RAG demo saved to: {demo_file}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.neo4j_db:
            self.neo4j_db.close()
        logger.info("🧹 Pipeline cleanup complete")

def main():
    """Main execution function."""
    
    print("🚀 NEO4J RAG PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = Neo4jRAGPipeline()
    
    try:
        # Setup components
        if not pipeline.setup_components():
            print("❌ Component setup failed")
            return
        
        # Process PDFs (start with a small number for testing)
        max_files = 5  # Process first 5 PDFs for testing
        
        stats = pipeline.process_pdf_directory(
            pdf_directory="pdfs",
            max_files=max_files,
            save_json=True
        )
        
        print(f"\n🎉 PIPELINE COMPLETE!")
        print(f"   Processed: {stats['processed']}")
        print(f"   Failed: {stats['failed']}")
        print(f"   Ingested to Neo4j: {stats['ingested']}")
        
        # Create RAG demo
        pipeline.create_rag_demo()
        
        print(f"\n📋 NEXT STEPS:")
        if stats['ingested'] > 0:
            print("   ✅ Neo4j ingestion successful")
            print("   🔍 Query your graph database")
        else:
            print("   📄 Data saved to JSON files")
            print("   🎯 Run: python neo4j_output/rag_demo.py")
        
        print(f"\n📊 Check neo4j_output/ directory for processed data")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        logger.debug(traceback.format_exc())
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main() 