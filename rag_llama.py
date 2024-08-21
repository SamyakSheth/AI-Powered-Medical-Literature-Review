import os
import re
import string
import ollama
import json
import pytesseract
import numpy as np
import pandas as pd
import concurrent.futures
from pdf2image import convert_from_path
from langchain.chains import RetrievalQA
from ragatouille import RAGPretrainedModel
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.document_loaders import PDFMinerLoader
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from llama_parse import LlamaParse
from langchain_community.embeddings import OllamaEmbeddings




class RAGBasedChatbot:
    def __init__(self,db_faiss_path):
         # Initialization of key variables and paths
        self.db_faiss_path = db_faiss_path
        self.loader = None
        self.docs = None
        self.texts = None
        self.embeddings = None
        self.index = None
        self.retriever = None
        self.llm = None
        self.prompt = None
        self.rag_chain = None
        self.responses = None
        self.compression_retriever = None
        self.temperature = 0
        self.format = 'json'
        self.num_gpu = 1
        self.num_predict=-2
        print("System Initailized! Hellooooooo")

    # def convert_pdf_to_images(self, pdf_path):
    #     # Converts PDF pages to images using pdf2image
    #     pages = convert_from_path(pdf_path)
    #     return pages
    
    # def get_conf(self, deskewed_image):
    #     # Calculate OCR confidence for the given image
    #     try:
    #         data = pytesseract.image_to_data(deskewed_image, output_type=pytesseract.Output.DICT)
    #         confidences = [int(conf) for conf in data['conf'] if conf != '-1']
    #         avg_confidence = np.mean(confidences) if confidences else 0
    #         return avg_confidence
    #     except Exception as e:
    #         print(f"Error calculating confidence: {e}")
    #         return -1

    # def clean_text(self, text):
    #     # Cleans the extracted text by removing unwanted characters
    #     allowed_punctuation = r'.,!?;:\-()\[\]%'
    #     pattern = f'[^{string.ascii_letters + string.digits + string.whitespace + allowed_punctuation}]'
    #     cleaned_text = re.sub(pattern, '', text)
    #     return cleaned_text.lower()

    # def process_page(self, page):
    #     # Processes each page to extract text and calculate OCR confidence
    #     page_conf = self.get_conf(page)
    #     d = pytesseract.image_to_data(page, output_type=pytesseract.Output.DICT)
    #     d_df = pd.DataFrame.from_dict(d)
    #     d_df = d_df[d_df.text != '']
    #     text = ' '.join(d_df.text)
    #     return page_conf, text

    def load_document(self, file_path):
        self.docs=[]

        ## Using LlamaParse
        
        api_key = 'llx-9hf1pg34vJphjGTS9gMvy5cPHffQQNG4ZvPbgAo1jYYcvSTg' #a
        #api_key = 'llx-uZfKgkE2M3WrEbyC3Ftxyc1BWlcVWOvhPAWRl8kpq52xI2SP' #b
        #api_key = 'llx-xJTD207urMCJMYyvLhh22fZnkAmj0M7CNWIbed1ITpmc22SC' #c
        parser = LlamaParse(
            api_key=api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
            result_type="markdown" # "markdown" and "text" are available
            #parsing_instruction = parsing_instructions
        )

        # sync
        d = parser.load_data(file_path)

        for ele in d: 
            self.docs.append(ele.text)

        #Uncomment the following if you want to use pytesseract OCR for loading data

        # #print("Loading PDF...")
        # pages = self.convert_pdf_to_images(file_path)
        # self.docs = []
        # #print("Pre-processing...")

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     results = list(executor.map(self.process_page, pages))

        # for _, text in results:
        #     cleaned_text = self.clean_text(text.strip())
        #     self.docs.append(cleaned_text)

        return len(self.docs)

    def find_header_footer_references(self, pages, file_path):
        # Identifies and extracts headers, footers, and references in the document
        loader = UnstructuredFileLoader(file_path, mode='elements', strategy='fast')
        d = loader.load()
        d.reverse()
        patterns = [r'\bReferences\b', r'\bBibliography\b', r'\bWorks Cited\b', r'\bCited References\b', 'References']
        references = None
        for doc in d:
            for pattern in patterns:
                if re.search(pattern, doc.page_content, re.IGNORECASE):
                    references = doc.metadata['page_number']
                    break
            if references is not None:
                break

        headers = []
        footers = []
        new = [ele for ele in d if ele.metadata['category'] in ['Header', 'Footer']]

        for i in range(pages):
            head = [ele for ele in new if ele.metadata['category'] == 'Header' and ele.metadata['page_number'] == i + 1]
            headers.append(self.clean_text(' '.join([x.page_content for x in head])) if head else "")

            foot = [ele for ele in new if ele.metadata['category'] == 'Footer' and ele.metadata['page_number'] == i + 1]
            footers.append(self.clean_text(' '.join([x.page_content for x in foot])) if foot else "")

        return headers, footers, references

    def preprocess(self, file_path):
        # Preprocesses the document to remove headers, footers, and references
        headers, footers, ref = self.find_header_footer_references(len(self.docs), file_path)
        self.docs = self.docs[:ref]
        headers = headers[:ref]
        footers = footers[:ref]
        for i in range(len(self.docs)):
            self.docs[i] = re.sub(re.escape(headers[i]), '', self.docs[i], flags=re.IGNORECASE).strip()
            self.docs[i] = re.sub(re.escape(footers[i]), '', self.docs[i], flags=re.IGNORECASE).strip()
        if ref is not None:
            patterns = [r'\bReferences\b', r'\bBibliography\b', r'\bWorks Cited\b', r'\bCited References\b']
            combined_pattern = '|'.join(patterns)
            try:
                split_text = re.split(combined_pattern, self.docs[ref - 1], flags=re.IGNORECASE)
            except:
                split_text = re.split(combined_pattern, self.docs[-1], flags=re.IGNORECASE)

            if len(split_text) > 1:
                self.docs[ref - 1] = split_text[0]

    def split_document(self):
        # Splits the cleaned document into chunks for better processing and embedding
        
        text_splitter = RecursiveCharacterTextSplitter.from_language("markdown", chunk_size= 500, chunk_overlap= 10)
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=10)
        
        self.texts = text_splitter.create_documents(self.docs)

        # semantic_chunker = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
        # self.texts = semantic_chunker.create_documents(self.docs)
        # print(self.texts)

    def setup_embedding(self):
        # Sets up the embeddings model

        # self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        self.embeddings = OllamaEmbeddings(model='nomic-embed-text')

    def create_vector_db(self):
        # Creates and saves a FAISS vector database from the embedded document chunks
        
        try:
            db = FAISS.from_documents(self.texts, self.embeddings)
            db.save_local(self.db_faiss_path)
        except: 
            return "Error Loading pdf"
    
    def setup_retriever(self):
        # Sets up a retriever using the FAISS vector database

        self.retriever = FAISS.from_documents(self.texts,self.embeddings).as_retriever(search_type="similarity",search_kwargs={"k": 10})
        #self.docsearch = FAISS.load_local(self.db_faiss_path, self.embeddings, allow_dangerous_deserialization=True)

    def setup_reranker(self):
        # Sets up a contextual retriever with reranking capabilities using RAG

        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=RAG.as_langchain_document_compressor(), base_retriever=self.retriever)


    def setup_llm(self,model_name = 'llama3'):  
        # Configures the LLM (Ollama) for generating responses make sure the model is pulled first in Ollama
        self.llm = ChatOllama(
            model=model_name,
            temperature=self.temperature,
            format=self.format,
            num_predict=self.num_predict
        )

    def create_prompt_template(self,title,abstract):
        # Creates a prompt template for the LLM to generate appropriate responses
    
        template = f"""Title: {title}\nAbstract: {abstract}\n"""
        system_prompt = '''You are a Medical Literature Review Expert. Based on the given context, answer the following question correctly and provide reasons to support your answer. The output should be in JSON format with the following fields:\n- \"answer\": Yes, No, or Unknown\n- \"reason\": A brief explanation for your decision'''
        template = template + system_prompt +"""\nContext: {context}\nQuestion: {input}"""
        self.prompt = PromptTemplate(template=template, input_variables= ["context","input"])
        #print(self.prompt)
        #print(self.prompt)

    def create_rag_chain(self):
        # Creates the RAG chain that combines document retrieval and language generation
        
        combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(self.compression_retriever, combine_docs_chain)
        # self.rag_chain = RetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     chain_type="retriver",
        #     retriever=self.compression_retriever,
        #     return_source_documents=True,
        #     chain_type_kwargs={'prompt': self.prompt}
        # )

    def ask_question(self,question): 
        # Queries the RAG chain to answer a given question based on the processed document
        return self.rag_chain.invoke({'input': question})
    