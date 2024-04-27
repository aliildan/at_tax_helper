
import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class PdfLoader:
    def __init__(self, document_path: str):
        """
        Initialize the DocumentService with a document path
        :param document_path:
        """
        self.validate_document_path(document_path)
        self.document_path = document_path
        self.document_loader = PyPDFLoader(file_path=document_path)
        try:
            self.document = self.document_loader.load()
        except Exception as e:
            logging.error(f"Failed to load document: {e}")
            raise

    def validate_document_path(self, path):
        if not path:
            raise ValueError("Document path is required")
        if not path.endswith('.pdf'):
            raise ValueError("Document path must be a PDF file")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Document does not exist at the path: {path}")

    def get_document(self):
        """
        Get the entire document
        :return:
        """
        return self.document

    def get_page_content(self, page_number: int):
        """
        Get the content of a page in the document
        :param page_number:
        :return:
        """
        if page_number > len(self.document):
            raise Exception("Page number is out of range")
        return self.document[page_number].page_content

    def get_page_count(self):
        """
        Get the number of pages in the document
        :return:
        """
        return len(self.document)

    def split_into_chunks(self, page_range: tuple = None, chunk_size: int = 7500, chunk_overlap: int = 100):
        """
        Split the document into chunks
        page_range is a tuple of start_page and end_page (both inclusive)
        chunk_size is the maximum number of characters in a chunk
        chunk_overlap is the number of characters that overlap between two consecutive chunks
        :param page_range:
        :param chunk_size:
        :param chunk_overlap:
        :return:
        """
        if page_range:
            start_page, end_page = page_range
            if start_page > len(self.document) or end_page > len(self.document):
                raise ValueError("Page number is out of range")
            document = self.document[start_page:end_page]
        else:
            document = self.document

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(document)
        return chunks

