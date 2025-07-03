#!/usr/bin/env python3
"""
Neo4j AuraDB Integration for RAG System

This module provides Neo4j cloud database integration.
"""

from neo4j import GraphDatabase
import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
from pathlib import Path
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jRAGDatabase:
    """Neo4j AuraDB manager for RAG system."""
    
    def __init__(self, uri: str, username: str = "neo4j", password: str = ""):
        """Initialize Neo4j AuraDB connection."""
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info("✅ Neo4j AuraDB connection established")
                else:
                    raise Exception("Connection test failed")
                    
        except Exception as e:
            logger.error(f"❌ Neo4j AuraDB connection failed: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j AuraDB connection closed")
    
    def setup_schema(self):
        """Create the complete Neo4j schema with constraints and indexes."""
        
        logger.info("🏗️ Setting up Neo4j AuraDB schema...")
        
        # Schema setup queries - Core node constraints
        constraint_queries = [
            "CREATE CONSTRAINT document_path IF NOT EXISTS FOR (d:Document) REQUIRE d.path IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE", 
            "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (con:Concept) REQUIRE con.name IS UNIQUE"
        ]
        
        # Performance indexes
        index_queries = [
            "CREATE INDEX document_filename IF NOT EXISTS FOR (d:Document) ON (d.filename)",
            "CREATE INDEX chunk_document_path IF NOT EXISTS FOR (c:Chunk) ON (c.document_path)",
            "CREATE INDEX person_name_idx IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX org_name_idx IF NOT EXISTS FOR (o:Organization) ON (o.name)"
        ]
        
        # Full-text search indexes
        fulltext_queries = [
            "CREATE FULLTEXT INDEX chunk_text_search IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
            "CREATE FULLTEXT INDEX document_search IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.subject, d.author]"
        ]
        
        with self.driver.session() as session:
            # Create constraints
            for query in constraint_queries:
                try:
                    session.run(query)
                    logger.debug(f"✅ Constraint: {query.split()[-4]}")
                except Exception as e:
                    logger.debug(f"⚠️ Constraint exists or failed: {e}")
            
            # Create indexes
            for query in index_queries:
                try:
                    session.run(query)
                    logger.debug(f"✅ Index: {query.split()[2]}")
                except Exception as e:
                    logger.debug(f"⚠️ Index exists or failed: {e}")
            
            # Create full-text indexes
            for query in fulltext_queries:
                try:
                    session.run(query)
                    logger.debug(f"✅ Full-text index: {query.split()[3]}")
                except Exception as e:
                    logger.debug(f"⚠️ Full-text index exists or failed: {e}")
        
        logger.info("✅ Neo4j AuraDB schema setup complete")
    
    def batch_ingest_document(self, document_data: Dict[str, Any], 
                            extraction_data: List[Dict[str, Any]] = None,
                            embeddings: List[List[float]] = None) -> Dict[str, int]:
        """Batch ingest complete document with all data."""
        
        # Handle DocumentMetadata object properly (could be string from JSON)
        document = document_data.get('document')
        if isinstance(document, str):
            # Extract filename from string representation
            import re
            match = re.search(r"filename='([^']+)'", document)
            doc_filename = match.group(1) if match else 'unknown'
        elif hasattr(document, 'filename'):
            doc_filename = document.filename
        elif isinstance(document, dict):
            doc_filename = document.get('filename', 'unknown')
        else:
            doc_filename = 'unknown'
        
        logger.info(f"📥 Ingesting to AuraDB: {doc_filename}")
        
        stats = {'documents': 0, 'chunks': 0, 'entities': 0, 'relationships': 0}
        
        try:
            with self.driver.session() as session:
                with session.begin_transaction() as tx:
                    
                    # 1. Create document
                    if 'document' in document_data:
                        document = document_data['document']
                        # Convert DocumentMetadata to dict properly
                        if isinstance(document, str):
                            # Parse string representation to extract key info
                            import re
                            doc_dict = {}
                            patterns = {
                                'path': r"path='([^']+)'",
                                'filename': r"filename='([^']+)'",
                                'title': r"title='([^']*)'",
                                'author': r"author='([^']*)'",
                                'page_count': r"page_count=(\d+)"
                            }
                            for key, pattern in patterns.items():
                                match = re.search(pattern, document)
                                if match:
                                    if key == 'page_count':
                                        doc_dict[key] = int(match.group(1))
                                    else:
                                        doc_dict[key] = match.group(1)
                                else:
                                    doc_dict[key] = '' if key != 'page_count' else 0
                        elif hasattr(document, '__dict__'):
                            doc_dict = asdict(document)
                        else:
                            doc_dict = document
                        doc_path = self._create_document_tx(tx, doc_dict)
                        stats['documents'] = 1
                    
                    # 2. Process chunks
                    chunks = document_data.get('chunks', [])
                    extraction_data = extraction_data or [{}] * len(chunks)
                    embeddings = embeddings or [None] * len(chunks)
                    
                    for i, chunk in enumerate(chunks):
                        # Convert TextChunk to dict properly
                        if isinstance(chunk, str):
                            # Parse string representation to extract key info
                            import re
                            chunk_dict = {}
                            patterns = {
                                'chunk_id': r"chunk_id='([^']+)'",
                                'text': r"text='([^']+)'",
                                'document_path': r"document_path='([^']+)'",
                                'word_count': r"word_count=(\d+)",
                                'char_count': r"char_count=(\d+)"
                            }
                            for key, pattern in patterns.items():
                                match = re.search(pattern, chunk)
                                if match:
                                    if key in ['word_count', 'char_count']:
                                        chunk_dict[key] = int(match.group(1))
                                    else:
                                        chunk_dict[key] = match.group(1)
                                else:
                                    chunk_dict[key] = '' if key not in ['word_count', 'char_count'] else 0
                        elif hasattr(chunk, '__dict__'):
                            chunk_dict = asdict(chunk)
                        else:
                            chunk_dict = chunk
                        
                        # Add embedding
                        if i < len(embeddings) and embeddings[i] is not None:
                            chunk_dict['embedding'] = embeddings[i]
                        
                        # Create chunk
                        chunk_id = self._create_chunk_tx(tx, chunk_dict)
                        stats['chunks'] += 1
                        
                        # Link to document
                        if 'document' in document_data:
                            self._link_document_chunk_tx(tx, doc_path, chunk_id)
                        
                        # Process entities and relationships
                        if i < len(extraction_data):
                            ie_data = extraction_data[i]
                            
                            # Create entities
                            for entity in ie_data.get('entities', []):
                                self._create_entity_tx(tx, entity, chunk_id)
                                stats['entities'] += 1
                            
                            # Create relationships
                            for relationship in ie_data.get('relationships', []):
                                if self._create_relationship_tx(tx, relationship):
                                    stats['relationships'] += 1
                    
                    logger.info(f"✅ Ingested to AuraDB {doc_filename}: {stats}")
                    return stats
                    
        except Exception as e:
            logger.error(f"❌ Error ingesting {doc_filename} to AuraDB: {e}")
            return stats
    
    # Transaction helpers
    def _create_document_tx(self, tx, doc_data: Dict[str, Any]) -> str:
        query = """
        MERGE (d:Document {path: $path})
        SET d.filename = $filename, d.title = $title, d.author = $author,
            d.page_count = $page_count, d.processed_date = datetime()
        RETURN d.path as path
        """
        result = tx.run(query, doc_data)
        return result.single()["path"]
    
    def _create_chunk_tx(self, tx, chunk_data: Dict[str, Any]) -> str:
        query = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.text = $text, c.document_path = $document_path,
            c.word_count = $word_count, c.embedding = $embedding
        RETURN c.chunk_id as chunk_id
        """
        result = tx.run(query, chunk_data)
        return result.single()["chunk_id"]
    
    def _link_document_chunk_tx(self, tx, doc_path: str, chunk_id: str):
        query = """
        MATCH (d:Document {path: $doc_path})
        MATCH (c:Chunk {chunk_id: $chunk_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        tx.run(query, {'doc_path': doc_path, 'chunk_id': chunk_id})
    
    def _create_entity_tx(self, tx, entity_data: Dict[str, Any], chunk_id: str):
        entity_type = entity_data.get('label', 'Entity')
        label_map = {'Person': 'Person', 'Organization': 'Organization', 
                    'Location': 'Location', 'PERSON': 'Person', 'ORG': 'Organization'}
        label = label_map.get(entity_type, 'Entity')
        
        query = f"""
        MERGE (e:{label} {{name: $name}})
        SET e.last_seen = datetime()
        WITH e
        MATCH (c:Chunk {{chunk_id: $chunk_id}})
        MERGE (c)-[:MENTIONS]->(e)
        """
        tx.run(query, {'name': entity_data.get('text', ''), 'chunk_id': chunk_id})
    
    def _create_relationship_tx(self, tx, rel_data: Dict[str, Any]) -> bool:
        query = """
        MATCH (e1) WHERE e1.name = $entity1
        MATCH (e2) WHERE e2.name = $entity2
        MERGE (e1)-[r:SEMANTIC_RELATION {type: $rel_type}]->(e2)
        SET r.confidence = $confidence
        RETURN count(r) as created
        """
        result = tx.run(query, {
            'entity1': rel_data.get('entity1', ''),
            'entity2': rel_data.get('entity2', ''),
            'rel_type': rel_data.get('relationship', 'RELATES_TO'),
            'confidence': rel_data.get('confidence', 0.8)
        })
        return result.single()["created"] > 0
    
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        
        stats_queries = [
            ("documents", "MATCH (d:Document) RETURN count(d) as count"),
            ("chunks", "MATCH (c:Chunk) RETURN count(c) as count"),
            ("people", "MATCH (p:Person) RETURN count(p) as count"),
            ("organizations", "MATCH (o:Organization) RETURN count(o) as count"),
            ("locations", "MATCH (l:Location) RETURN count(l) as count"),
            ("concepts", "MATCH (c:Concept) RETURN count(c) as count"),
            ("relationships", "MATCH ()-[r:SEMANTIC_RELATION]->() RETURN count(r) as count")
        ]
        
        stats = {}
        with self.driver.session() as session:
            for name, query in stats_queries:
                try:
                    result = session.run(query)
                    stats[name] = result.single()["count"]
                except:
                    stats[name] = 0
        
        return stats
    
    def store_guardian_relationships(self, relationships: List[Any]) -> int:
        """Store Guardian Brief relationships in Neo4j."""
        logger.info(f"💾 Storing {len(relationships)} Guardian Brief relationships")
        
        stored_count = 0
        
        try:
            with self.driver.session() as session:
                with session.begin_transaction() as tx:
                    for rel in relationships:
                        # Convert relationship object to dict if needed
                        if hasattr(rel, '__dict__'):
                            rel_dict = rel.__dict__
                        else:
                            rel_dict = rel
                        
                        # Create entities and relationship based on Guardian Brief schema
                        entity1_name = rel_dict.get('entity1', '')
                        entity2_name = rel_dict.get('entity2', '')
                        relationship_type = rel_dict.get('relationship_type', 'RELATES_TO')
                        
                        if not entity1_name or not entity2_name:
                            continue
                        
                        # Determine entity types based on Guardian Brief ontology
                        entity1_type = self._determine_guardian_entity_type(entity1_name)
                        entity2_type = self._determine_guardian_entity_type(entity2_name)
                        
                        # Create entities with appropriate labels
                        query_create_entities = f"""
                        MERGE (e1:{entity1_type} {{name: $entity1}})
                        SET e1.last_seen = datetime()
                        MERGE (e2:{entity2_type} {{name: $entity2}})
                        SET e2.last_seen = datetime()
                        """
                        
                        tx.run(query_create_entities, {
                            'entity1': entity1_name,
                            'entity2': entity2_name
                        })
                        
                        # Create relationship with Guardian Brief properties
                        query_create_relationship = f"""
                        MATCH (e1:{entity1_type} {{name: $entity1}})
                        MATCH (e2:{entity2_type} {{name: $entity2}})
                        MERGE (e1)-[r:{relationship_type}]->(e2)
                        SET r.confidence = $confidence,
                            r.legal_context = $legal_context,
                            r.evidence = $evidence,
                            r.jurisdiction = $jurisdiction,
                            r.precedent_strength = $precedent_strength,
                            r.created_date = datetime()
                        """
                        
                        tx.run(query_create_relationship, {
                            'entity1': entity1_name,
                            'entity2': entity2_name,
                            'confidence': rel_dict.get('confidence', 0.8),
                            'legal_context': rel_dict.get('legal_context', ''),
                            'evidence': rel_dict.get('evidence', ''),
                            'jurisdiction': rel_dict.get('jurisdiction', ''),
                            'precedent_strength': rel_dict.get('precedent_strength', '')
                        })
                        
                        stored_count += 1
                        
            logger.info(f"✅ Stored {stored_count} Guardian Brief relationships")
            return stored_count
            
        except Exception as e:
            logger.error(f"❌ Error storing Guardian Brief relationships: {e}")
            return stored_count
    
    def _determine_guardian_entity_type(self, entity_name: str) -> str:
        """Determine entity type based on Guardian Brief ontology."""
        entity_lower = entity_name.lower()
        
        # Legal entities
        if any(term in entity_lower for term in ['court', 'constitution', 'law', 'act', 'statute', 'regulation']):
            return 'LegalInstrument'
        
        # Natural entities
        if any(term in entity_lower for term in ['river', 'forest', 'mountain', 'lake', 'ocean', 'ecosystem', 'nature']):
            return 'NaturalEntity'
        
        # Organizations
        if any(term in entity_lower for term in ['government', 'ministry', 'department', 'agency', 'company', 'corporation']):
            return 'Organization'
        
        # Legal concepts
        if any(term in entity_lower for term in ['rights', 'personhood', 'standing', 'guardianship', 'protection']):
            return 'LegalConcept'
        
        # Locations
        if any(term in entity_lower for term in ['country', 'state', 'province', 'city', 'region']):
            return 'Location'
        
        # Default to LegalConcept for Guardian Brief context
        return 'LegalConcept'

# Create alias for backward compatibility
Neo4jDatabase = Neo4jRAGDatabase

if __name__ == "__main__":
    print("🗃️ Neo4j AuraDB Integration Test")
    print("=" * 40)
    
    # Load credentials from .env2
    from dotenv import load_dotenv
    load_dotenv('.env2')
    
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    if not neo4j_uri or not neo4j_password:
        print("❌ Neo4j credentials not found in .env2")
        exit(1)
    
    try:
        # Test connection
        db = Neo4jRAGDatabase(neo4j_uri, neo4j_username, neo4j_password)
        
        # Setup schema
        db.setup_schema()
        
        # Get stats
        stats = db.get_statistics()
        print("\n📊 AuraDB Statistics:")
        for key, value in stats.items():
            print(f"   {key.capitalize()}: {value}")
        
        db.close()
        print("\n✅ Neo4j AuraDB ready for RAG ingestion!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n🔧 Check your .env2 credentials") 