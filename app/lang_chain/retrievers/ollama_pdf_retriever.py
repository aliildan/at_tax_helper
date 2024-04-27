from langchain.retrievers import MultiQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.lang_chain.embeddings.ollama_embedding import OllamaEmbedding


class OllamaPdfRetriever:
    PROMPT_DOC_QUESTION_TEMPLATE = """Your task is given user question to retrieve relevant documents from a
     vector database. The user question is: {{question}}"""
    CHAT_DOC_QUESTION_TEMPLATE = """Answer the question based ONLY on the following context: {context}
    Question: {question}"""

    def __init__(self, model="llama3"):
        self.embedding = OllamaEmbedding(model=model)
        self.chat_ollama = ChatOllama(model=model)

    def get_chat_prompt_template(self, template: str):
        if template is None:
            template = self.CHAT_DOC_QUESTION_TEMPLATE
        return ChatPromptTemplate.from_template(template)

    def get_doc_prompt_template(self, template: str) -> PromptTemplate:
        if template is None:
            template = self.PROMPT_DOC_QUESTION_TEMPLATE
        return PromptTemplate(input_variables=["question"], template=template)

    def get_retriever(self, collection_name: str, prompt_template: str = None) -> MultiQueryRetriever:
        vector_as_retriever = self.embedding.get_db(collection_name).as_retriever(search_type="mmr")
        return MultiQueryRetriever.from_llm(retriever=vector_as_retriever, llm=self.chat_ollama,
                                            prompt=self.get_doc_prompt_template(prompt_template))

    def get_chain(self, collection_name: str, chat_prompt_template: str = None):
        retriever = self.get_retriever(collection_name)
        prompt = self.get_chat_prompt_template(chat_prompt_template)
        llm = self.chat_ollama
        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        return chain

