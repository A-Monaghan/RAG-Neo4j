# RAG (Retrieval-Augmented Generation) Project

This project ingests PDFs, extracts and cleans entities and relationships, generates embeddings, and loads the results into a Neo4j graph database for Retrieval-Augmented Generation (RAG) and advanced knowledge graph applications.

---

## Project Overview

- **Input:** PDFs (in `pdfs/`)
- **Pipeline:**
  1. PDF text extraction and chunking
  2. Entity and relationship extraction (NER, LLMs)
  3. Data cleaning and normalization
  4. Embedding generation
  5. Neo4j graph ingestion (nodes, relationships)
  6. RAG demo and Cypher query interface
- **Output:**
  - Cleaned JSONs, Neo4j-ready data, and a live knowledge graph

---

## Current Status (June 2024)

- **PDF → Neo4j pipeline is operational** (see `run_neo4j_pipeline.py` and `complete_neo4j_pipeline.py`)
- **Entity/relationship extraction**: Multiple extractors (LLM, rule-based, etc.)
- **Data cleaning**: Advanced cleaning scripts in place
- **Neo4j integration**: Supports both cloud and local/AuraDB
- **Batch and incremental processing**: Process all or only new PDFs
- **Dashboards and reports**: Markdown and PNG dashboards for analysis
- **Testing**: Multiple test scripts for ingestion, extraction, and cleaning
- **Next:**
  - Expand RAG query/demo interface
  - Improve relationship extraction and graph analytics
  - Enhance embedding and semantic search

---

## Directory Structure (Key Folders)

```
RAG/
├── pdfs/                   # Input PDFs
├── output/                 # Processed output data
├── guardian_extractions/   # Extracted relationships/entities (JSON)
├── neo4j_output/           # Neo4j-ready outputs
├── cleaning/               # Data cleaning scripts
├── processing/             # Chunking, document processing
├── extraction/             # Entity/relationship extraction
├── graph/                  # Graph schema/utilities
├── src/                    # Main source code (API, embeddings, RAG logic)
├── config/                 # Config files (YAML, env)
├── examples/               # Example scripts
├── venv/                   # Python virtual environment (ignored)
├── requirements.txt        # Python dependencies
├── .env2                   # Environment variables (not committed)
├── README.md               # This file
└── ...                     # See full repo for all scripts
```

---

## Main Scripts & Usage

### 1. **Full Pipeline (Cloud/Local Neo4j)**

- **run_neo4j_pipeline.py**: Main entry point (uses `.env2` for credentials)
- **run_neo4j_with_direct_creds.py**: For direct Neo4j credentials
- **complete_neo4j_pipeline.py**: Orchestrates the full workflow (can be run directly)

**To run the main pipeline:**
```bash
# Activate your virtual environment first
python run_neo4j_pipeline.py
```
- Adjust `max_files` in the script to process more/fewer PDFs.
- Logs and progress: see `neo4j_rag_pipeline.log` and output folders.

### 2. **Incremental/Partial Processing**
- **process_remaining_pdfs.py**: Processes only unprocessed PDFs, analyzes data
- **simple_pdf_processing.py**: Basic PDF chunking (for dev/testing)
- **test_*.py**: Test scripts for various modules

---

## Features

| Feature                        | Status         |
|------------------------------- |--------------- |
| PDF text extraction            | ✅ Complete    |
| Text chunking                  | ✅ Complete    |
| Entity extraction (NER/LLM)    | ✅ Complete    |
| Relationship extraction        | ✅ Complete    |
| Data cleaning                  | ✅ Complete    |
| Embedding generation           | ✅ Complete    |
| Neo4j ingestion                | ✅ Complete    |
| Batch/incremental processing   | ✅ Complete    |
| RAG demo/query interface       | ⏳ In progress |
| Dashboards/analysis            | ✅ Complete    |
| Testing                        | ✅ Complete    |

---

## Outputs & Next Steps

- **Outputs:**
  - Cleaned and processed JSONs (see `output/`, `guardian_extractions/`, `neo4j_output/`)
  - Neo4j knowledge graph (cloud/local)
  - Dashboards and analysis reports
- **Next Steps:**
  - Expand RAG query/demo interface
  - Enhance analytics and relationship extraction
  - Integrate more advanced semantic search
  - **Explore advanced chunking strategies for RAG:**
    - See [ALucek/chunking-strategies](https://github.com/ALucek/chunking-strategies) for the latest research and practical implementations.
    - Techniques to consider:
      - Character/Token Based Chunking
      - Recursive Character/Token Based Chunking
      - Semantic Chunking
      - Cluster Semantic Chunking
      - LLM Semantic Chunking
    - These methods can optimize how documents are split for embedding and retrieval, potentially improving downstream RAG performance.

---

## Requirements

- Python 3.8+
- Neo4j (AuraDB or local)
- See `requirements.txt` and `neo4j_requirements.txt`

---

## Legacy/Basic PDF Processing

The original pipeline (now superseded by the full pipeline above) is in `pdf_processor.py`:

- **extract_text_from_pdf(pdf_path)**: Extracts text from a single PDF file
- **clean_text(text)**: Cleans and preprocesses extracted text
- **chunk_text(text, chunk_size=1000, overlap=200)**: Splits text into chunks
- **process_pdf_directory(pdf_dir="pdfs")**: Processes all PDFs in a directory

**To run the basic processor:**
```bash
python pdf_processor.py
```

---

## Contributing

This is an active, evolving project. Contributions and suggestions are welcome!

- See `NEO4J_RAG_ROADMAP.md` and `FINAL_STATUS_REPORT.md` for more details on progress and plans. # RAG-Neo4j
