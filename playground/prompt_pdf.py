from app.lang_chain.retrievers.ollama_pdf_retriever import OllamaPdfRetriever

retriever = OllamaPdfRetriever()
chain = retriever.get_chain("tax_books")
prompt = input("How can I help you today? ")
print(f"Please wait... Prompt : , '{prompt}'")
for chunk in chain.stream(prompt):
    print(chunk, end="", flush=True)