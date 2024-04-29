# A Basic LangChain, Ollama-LLAMA3 and Chroma Project


## Introduction
Welcome to **at_tax_helper**. This repository is a simple project demonstrating the basics of using LangChain, Ollama, Chroma, and Llama3 to interact with PDF documents containing Austrian tax guidelines.
Please check out LangChain's offical documentation for more information: 
[here](https://python.langchain.com/docs/get_started/introduction)
## Project Description)
This project utilizes LangChain to create a dialogue system for interacting with PDF files, specifically tax guides from AK and Finanzamt. The guides are indexed using Chroma with Ollama embeddings for efficient querying. It's a basic implementation, meant to introduce the possibilities of LangChain and is ready for further development and more complex integrations.

## Key Technologies
- **LangChain**: For building the dialogue system.
- **Ollama and Llama3**: Advanced language models used in processing and understanding the documents.
- **Chroma**: For indexing documents with vector-based search capabilities.

## Requirements
This project runs on Python versions 3.8 to 3.11. Ensure you have Python installed in your environment before proceeding. All dependencies needed to run this project are listed in the `requirements.txt` file.

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/aliildan/at_tax_helper.git
   cd at_tax_helper


2. **Install dependencies:**:
   ```bash
   pip install -r requirements.txt

3. **Install Ollama and run LLAMA3:**:
   
   [ollama](https://ollama.com/library/llama3)
   ```bash
   ollama run llama3

4. **Usage:** :

      - `/plaground/import_pdfs.py` : Indexes the pdfs
      - `/playground/prmpt_pdf.py`  : A basic terminal to use LLM with prompts.
