from app.lang_chain.document_loaders.pdf_loader import PdfLoader
from app.lang_chain.embeddings.ollama_embedding import OllamaEmbedding

import os

directory_path = "./../data/pdfs/"
files = []
for root, dirs, filenames in os.walk(directory_path):
    for filename in filenames:
        files.append(os.path.join(directory_path, filename))
print(files)

embedding = OllamaEmbedding()

for file in files:
    print(file)
    pdf_loader = PdfLoader(file)
    documents = pdf_loader.split_into_chunks(chunk_size=5000, chunk_overlap=100)
    embedding.create_collection(collection_name="tax_books")
    embedding.save_documents(collection_name="tax_books", documents=documents)

print(embedding.get_collection(collection_name="tax_books").peek(limit=2))
