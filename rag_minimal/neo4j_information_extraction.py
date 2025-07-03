#!/usr/bin/env python3
"""
Information Extraction for Neo4j RAG System

This module provides:
1. Named Entity Recognition (NER) with spaCy
2. Relationship Extraction with Claude API
3. Keyword/Concept extraction with YAKE
4. Structured data for Neo4j ingestion
"""

import spacy
import os
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
from anthropic import Anthropic

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Named entity with metadata."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    chunk_id: str = ""

@dataclass
class Relationship:
    """Extracted relationship between entities."""
    entity1: str
    type1: str
    relationship: str
    entity2: str
    type2: str
    chunk_id: str = ""
    confidence: float = 1.0

@dataclass 
class Concept:
    """Extracted concept or keyword."""
    text: str
    score: float
    chunk_id: str = ""

# Define allowed entity types for Neo4j schema
ALLOWED_ENTITY_TYPES = {
    'PERSON': 'Person',
    'ORG': 'Organization', 
    'GPE': 'Location',  # Geopolitical entity
    'LOC': 'Location',
    'DATE': 'Date',
    'LAW': 'Legislation',
    'MONEY': 'Financial',
    'PERCENT': 'Financial',
    # Custom types for legal/environmental documents
    'LEGAL_CASE': 'LegalCase',
    'LEGISLATION': 'Legislation',
    'CONCEPT': 'Concept',
    'DEVELOPMENT_PROPOSAL': 'DevelopmentProposal',
    'PRINCIPLE': 'Principle'
}

# Define allowed relationship types
ALLOWED_RELATIONSHIPS = [
    'PROPOSED_BY',
    'CHALLENGED_BY', 
    'GRANTED_PERSONHOOD_TO',
    'CITED_IN',
    'IMPACTS',
    'LOCATED_IN',
    'AUTHORED_BY',
    'MENTIONS',
    'RELATES_TO',
    'PART_OF'
]

class NERProcessor:
    """Named Entity Recognition processor using spaCy."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize spaCy NER processor."""
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"spaCy model {model_name} not found. Install with: python -m spacy download {model_name}")
            raise
    
    def extract_entities(self, text: str, chunk_id: str = "") -> List[Entity]:
        """Extract named entities from text."""
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                # Map spaCy labels to our schema
                entity_type = ALLOWED_ENTITY_TYPES.get(ent.label_, ent.label_)
                
                entity = Entity(
                    text=ent.text.strip(),
                    label=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0,  # spaCy doesn't provide confidence scores by default
                    chunk_id=chunk_id
                )
                entities.append(entity)
            
            # Filter out very short or common entities
            entities = [e for e in entities if len(e.text) > 2 and e.text.lower() not in ['the', 'and', 'or', 'but']]
            
            logger.debug(f"Extracted {len(entities)} entities from chunk {chunk_id}")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

class RelationshipExtractor:
    """LLM-based relationship extraction using Claude."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude API client."""
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            logger.warning("No Claude API key provided. Relationship extraction will be disabled.")
            self.client = None
        else:
            self.client = Anthropic(api_key=self.api_key)
            logger.info("Claude API client initialized")
    
    def extract_relationships_with_claude(self, chunk_text: str, chunk_id: str = "") -> List[Relationship]:
        """Extract relationships using Claude API."""
        
        if not self.client:
            logger.warning("Claude API not available. Skipping relationship extraction.")
            return []
        
        try:
            # Construct the detailed prompt
            prompt = self._build_extraction_prompt(chunk_text)
            
            # Call Claude API
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Fast, cost-effective model
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Parse response
            relationships = self._parse_claude_response(response.content[0].text, chunk_id)
            
            logger.debug(f"Extracted {len(relationships)} relationships from chunk {chunk_id}")
            return relationships
            
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return []
    
    def _build_extraction_prompt(self, text: str) -> str:
        """Build the Claude prompt for relationship extraction."""
        
        prompt = f"""You are an expert at extracting structured relationships from legal, environmental, and policy documents.

ENTITY TYPES:
- Person: Individual people (authors, officials, etc.)
- Organization: Companies, government bodies, NGOs
- Location: Places, regions, countries, landmarks
- LegalCase: Court cases, legal proceedings
- Legislation: Laws, acts, regulations, policies
- Concept: Abstract ideas, principles, theories
- DevelopmentProposal: Proposed projects, developments
- Principle: Legal or environmental principles

RELATIONSHIP TYPES:
- PROPOSED_BY: X was proposed by Y
- CHALLENGED_BY: X was challenged by Y  
- GRANTED_PERSONHOOD_TO: X granted legal personhood to Y
- CITED_IN: X was cited in Y
- IMPACTS: X impacts/affects Y
- LOCATED_IN: X is located in Y
- AUTHORED_BY: X was authored by Y
- MENTIONS: X mentions Y
- RELATES_TO: X relates to Y

TASK: Extract relationships from this text and return ONLY a valid JSON list in this exact format:
[{{"entity1": "EntityName1", "type1": "EntityType", "relationship": "RELATIONSHIP_TYPE", "entity2": "EntityName2", "type2": "EntityType"}}]

TEXT TO ANALYZE:
{text}

RESPONSE (JSON only):"""
        
        return prompt
    
    def _parse_claude_response(self, response: str, chunk_id: str) -> List[Relationship]:
        """Parse Claude's JSON response into Relationship objects."""
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in Claude response")
                return []
            
            json_str = json_match.group(0)
            relationships_data = json.loads(json_str)
            
            relationships = []
            for rel_data in relationships_data:
                # Validate required fields
                required_fields = ['entity1', 'type1', 'relationship', 'entity2', 'type2']
                if not all(field in rel_data for field in required_fields):
                    logger.warning(f"Invalid relationship data: {rel_data}")
                    continue
                
                # Validate entity types and relationships
                type1 = rel_data['type1']
                type2 = rel_data['type2']
                relationship = rel_data['relationship']
                
                if relationship not in ALLOWED_RELATIONSHIPS:
                    logger.warning(f"Unknown relationship type: {relationship}")
                    continue
                
                rel = Relationship(
                    entity1=rel_data['entity1'],
                    type1=type1,
                    relationship=relationship,
                    entity2=rel_data['entity2'],
                    type2=type2,
                    chunk_id=chunk_id,
                    confidence=0.8  # Default confidence for LLM extraction
                )
                relationships.append(rel)
            
            return relationships
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Claude JSON response: {e}")
            logger.debug(f"Response was: {response}")
            return []
        except Exception as e:
            logger.error(f"Error processing Claude response: {e}")
            return []

class ConceptExtractor:
    """Extract key concepts and keywords using YAKE."""
    
    def __init__(self):
        """Initialize YAKE keyword extractor."""
        try:
            import yake
            self.kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,  # Extract up to 3-gram phrases
                dedupLim=0.7,
                top=10  # Top 10 keywords per chunk
            )
            logger.info("YAKE keyword extractor initialized")
        except ImportError:
            logger.warning("YAKE not available. Install with: pip install yake")
            self.kw_extractor = None
    
    def extract_concepts(self, text: str, chunk_id: str = "") -> List[Concept]:
        """Extract key concepts and keywords."""
        
        if not self.kw_extractor:
            # Fallback to simple frequency-based extraction
            return self._simple_keyword_extraction(text, chunk_id)
        
        try:
            keywords = self.kw_extractor.extract_keywords(text)
            
            concepts = []
            for score, keyword in keywords:
                # YAKE returns lower scores for better keywords
                # Convert to 0-1 scale where higher is better
                normalized_score = max(0, 1 - (score / 10))
                
                concept = Concept(
                    text=keyword,
                    score=normalized_score,
                    chunk_id=chunk_id
                )
                concepts.append(concept)
            
            logger.debug(f"Extracted {len(concepts)} concepts from chunk {chunk_id}")
            return concepts
            
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return self._simple_keyword_extraction(text, chunk_id)
    
    def _simple_keyword_extraction(self, text: str, chunk_id: str) -> List[Concept]:
        """Simple fallback keyword extraction."""
        
        # Simple approach: extract noun phrases using spaCy
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            
            # Extract noun chunks
            concepts = []
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 3 and chunk.text.lower() not in ['this', 'that', 'these', 'those']:
                    concept = Concept(
                        text=chunk.text.strip(),
                        score=0.5,  # Default score
                        chunk_id=chunk_id
                    )
                    concepts.append(concept)
            
            # Remove duplicates and limit
            seen = set()
            unique_concepts = []
            for concept in concepts:
                if concept.text.lower() not in seen and len(unique_concepts) < 10:
                    seen.add(concept.text.lower())
                    unique_concepts.append(concept)
            
            return unique_concepts
            
        except Exception as e:
            logger.error(f"Fallback concept extraction failed: {e}")
            return []

class InformationExtractor:
    """Main information extraction coordinator."""
    
    def __init__(self, claude_api_key: Optional[str] = None):
        """Initialize all extractors."""
        self.ner_processor = NERProcessor()
        self.relationship_extractor = RelationshipExtractor(claude_api_key)
        self.concept_extractor = ConceptExtractor()
        
        logger.info("Information extraction pipeline initialized")
    
    def process_chunk(self, chunk_text: str, chunk_id: str) -> Dict[str, Any]:
        """Process a single text chunk and extract all information."""
        
        logger.debug(f"Processing chunk {chunk_id}")
        
        try:
            # Extract entities
            entities = self.ner_processor.extract_entities(chunk_text, chunk_id)
            
            # Extract relationships (only if we have entities)
            relationships = []
            if entities and len(chunk_text) > 100:  # Only for substantial chunks
                relationships = self.relationship_extractor.extract_relationships_with_claude(chunk_text, chunk_id)
            
            # Extract concepts
            concepts = self.concept_extractor.extract_concepts(chunk_text, chunk_id)
            
            result = {
                'chunk_id': chunk_id,
                'entities': [asdict(e) for e in entities],
                'relationships': [asdict(r) for r in relationships],
                'concepts': [asdict(c) for c in concepts],
                'stats': {
                    'entity_count': len(entities),
                    'relationship_count': len(relationships),
                    'concept_count': len(concepts)
                }
            }
            
            logger.debug(f"Processed chunk {chunk_id}: {len(entities)} entities, {len(relationships)} relationships, {len(concepts)} concepts")
            return result
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            return {
                'chunk_id': chunk_id,
                'entities': [],
                'relationships': [],
                'concepts': [],
                'stats': {'entity_count': 0, 'relationship_count': 0, 'concept_count': 0}
            }

def process_document_chunks(chunks: List[Any], claude_api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Process multiple chunks from a document."""
    
    extractor = InformationExtractor(claude_api_key)
    results = []
    
    logger.info(f"Processing {len(chunks)} chunks for information extraction")
    
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Processing chunk {i}/{len(chunks)}: {chunk.chunk_id}")
        
        try:
            result = extractor.process_chunk(chunk.text, chunk.chunk_id)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
            # Add empty result to maintain order
            results.append({
                'chunk_id': chunk.chunk_id,
                'entities': [],
                'relationships': [],
                'concepts': [],
                'stats': {'entity_count': 0, 'relationship_count': 0, 'concept_count': 0}
            })
    
    # Summary statistics
    total_entities = sum(r['stats']['entity_count'] for r in results)
    total_relationships = sum(r['stats']['relationship_count'] for r in results)
    total_concepts = sum(r['stats']['concept_count'] for r in results)
    
    logger.info(f"Information extraction complete: {total_entities} entities, {total_relationships} relationships, {total_concepts} concepts")
    
    return results

if __name__ == "__main__":
    # Test information extraction
    print("🧠 Information Extraction for Neo4j RAG")
    print("=" * 50)
    
    # Test with sample text
    sample_text = """
    The Loch Lomond and The Trossachs National Park Authority has proposed new development guidelines 
    to protect the environmental integrity of the region. The proposal was challenged by Flamingo Land, 
    a development company seeking to build a large resort complex. Environmental groups argue that 
    granting legal personhood to natural ecosystems, as seen in New Zealand's Whanganui River case, 
    could provide stronger protection for Loch Lomond's unique biodiversity.
    """
    
    # Initialize extractor (without Claude API for testing)
    extractor = InformationExtractor()
    
    result = extractor.process_chunk(sample_text, "test_chunk_001")
    
    print(f"\n📊 EXTRACTION RESULTS:")
    print(f"   Entities: {result['stats']['entity_count']}")
    print(f"   Relationships: {result['stats']['relationship_count']}")
    print(f"   Concepts: {result['stats']['concept_count']}")
    
    print(f"\n🏷️  ENTITIES:")
    for entity in result['entities'][:5]:  # Show first 5
        print(f"   {entity['text']} ({entity['label']})")
    
    print(f"\n🔗 RELATIONSHIPS:")
    for rel in result['relationships'][:3]:  # Show first 3
        print(f"   {rel['entity1']} --{rel['relationship']}--> {rel['entity2']}")
    
    print(f"\n💡 CONCEPTS:")
    for concept in result['concepts'][:5]:  # Show first 5
        print(f"   {concept['text']} (score: {concept['score']:.2f})") 