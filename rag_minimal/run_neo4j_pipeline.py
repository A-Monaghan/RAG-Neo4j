#!/usr/bin/env python3
"""
Run Neo4j RAG Pipeline with Cloud Credentials

This script uses your .env2 credentials to process PDFs into Neo4j AuraDB.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env2
load_dotenv('.env2')

# Import our pipeline
from complete_neo4j_pipeline import Neo4jRAGPipeline

def main():
    """Run the complete Neo4j RAG pipeline with cloud credentials."""
    
    print("🚀 NEO4J CLOUD RAG PIPELINE")
    print("=" * 60)
    
    # Get credentials from environment
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    claude_api_key = os.getenv('ANTHROPIC_API_KEY')  # Optional
    
    print(f"🔗 Neo4j URI: {neo4j_uri}")
    print(f"👤 Username: {neo4j_username}")
    print(f"🔑 Password: {'*' * len(neo4j_password) if neo4j_password else 'Not set'}")
    print(f"🤖 Claude API: {'Available' if claude_api_key else 'Not available (will skip relationship extraction)'}")
    
    if not neo4j_uri or not neo4j_password:
        print("❌ Neo4j credentials not found in .env2")
        return
    
    # Initialize pipeline with cloud credentials
    pipeline = Neo4jRAGPipeline(
        claude_api_key=claude_api_key,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_username,
        neo4j_password=neo4j_password
    )
    
    try:
        # Setup components
        print("\n🔧 Setting up pipeline components...")
        if not pipeline.setup_components():
            print("❌ Component setup failed")
            return
        
        # Process PDFs - start with 5 for testing
        print("\n📁 Processing PDFs...")
        stats = pipeline.process_pdf_directory(
            pdf_directory="pdfs",
            max_files=5,  # Start with 5 PDFs for testing
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
            print("   ✅ Neo4j AuraDB ingestion successful!")
            print("   🌐 Access your graph at: https://console.neo4j.io/")
            print("   🔍 Run Cypher queries to explore your data")
        else:
            print("   📄 Data saved to JSON files in neo4j_output/")
            print("   🎯 Run: venv/bin/python neo4j_output/rag_demo.py")
        
        # Show some sample Cypher queries
        if stats['ingested'] > 0:
            print(f"\n🔍 SAMPLE CYPHER QUERIES:")
            print("   # Count all nodes")
            print("   MATCH (n) RETURN labels(n) as NodeType, count(n) as Count")
            print()
            print("   # Find documents and their chunks")
            print("   MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)")
            print("   RETURN d.filename, count(c) as ChunkCount")
            print()
            print("   # Find entities mentioned in chunks")
            print("   MATCH (c:Chunk)-[:MENTIONS]->(e)")
            print("   RETURN labels(e)[0] as EntityType, e.name, count(c) as MentionCount")
            print("   ORDER BY MentionCount DESC LIMIT 10")
        
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main() 