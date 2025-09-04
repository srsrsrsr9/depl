# langchain_foundation.py
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, PDFMinerLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.tools import Tool

from config import Config

class PharmaDataIngestion:
    """Handles ingestion of pharmaceutical data from various sources"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        
    def scrape_fda_guidance(self, drug_class: str) -> List[Document]:
        """Scrape FDA guidance documents for specific drug classes"""
        search_url = f"{Config.FDA_BASE_URL}/search/?s={drug_class}+guidance"
        
        try:
            response = requests.get(search_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            documents = []
            for link in soup.find_all('a', href=True):
                if 'guidance' in link.get('href', '').lower():
                    doc_url = link.get('href')
                    if not doc_url.startswith('http'):
                        doc_url = Config.FDA_BASE_URL + doc_url
                    
                    try:
                        doc_response = requests.get(doc_url)
                        doc_soup = BeautifulSoup(doc_response.content, 'html.parser')
                        content = doc_soup.get_text(strip=True)
                        
                        if len(content) > 500:  # Filter out small/irrelevant pages
                            documents.append(Document(
                                page_content=content,
                                metadata={
                                    "source": doc_url,
                                    "type": "fda_guidance",
                                    "drug_class": drug_class
                                }
                            ))
                    except Exception as e:
                        print(f"Error processing {doc_url}: {e}")
                        continue
                        
            return documents
        except Exception as e:
            print(f"Error scraping FDA guidance: {e}")
            return []
    
    def get_clinical_trials_data(self, condition: str, limit: int = 50) -> List[Document]:
        """Fetch clinical trials data from ClinicalTrials.gov API"""
        api_url = f"{Config.CLINICALTRIALS_API}/query/full_studies"
        params = {
            'expr': condition,
            'max_rnk': limit,
            'fmt': 'json'
        }
        
        try:
            response = requests.get(api_url, params=params)
            data = response.json()
            
            documents = []
            for study in data.get('FullStudiesResponse', {}).get('FullStudies', []):
                study_info = study.get('Study', {})
                protocol = study_info.get('ProtocolSection', {})
                
                # Extract key information
                title = protocol.get('IdentificationModule', {}).get('BriefTitle', '')
                description = protocol.get('DescriptionModule', {}).get('DetailedDescription', '')
                status = protocol.get('StatusModule', {}).get('OverallStatus', '')
                phase = protocol.get('DesignModule', {}).get('PhaseList', {}).get('Phase', ['Unknown'])
                
                content = f"Title: {title}\n\nDescription: {description}\n\nStatus: {status}\n\nPhase: {', '.join(phase) if isinstance(phase, list) else phase}"
                
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": "ClinicalTrials.gov",
                        "type": "clinical_trial",
                        "condition": condition,
                        "status": status,
                        "phase": phase
                    }
                ))
                
            return documents
        except Exception as e:
            print(f"Error fetching clinical trials: {e}")
            return []

class PharmaKnowledgeBase:
    """Vector store and retrieval system for pharmaceutical knowledge"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.vector_store = None
        self.ingestion = PharmaDataIngestion()
        
    def build_knowledge_base(self, drug_class: str, condition: str):
        """Build comprehensive knowledge base for a drug/condition"""
        print("Gathering FDA guidance documents...")
        fda_docs = self.ingestion.scrape_fda_guidance(drug_class)
        
        print("Fetching clinical trials data...")
        trial_docs = self.ingestion.get_clinical_trials_data(condition)
        
        # Combine all documents
        all_docs = fda_docs + trial_docs
        
        if not all_docs:
            print("No documents found. Using sample data...")
            # Fallback sample data for demonstration
            all_docs = [
                Document(
                    page_content="FDA guidance for oncology drug development requires robust clinical trial data demonstrating safety and efficacy.",
                    metadata={"source": "sample", "type": "guidance"}
                ),
                Document(
                    page_content="Competitive landscape analysis shows increasing investment in immunotherapy for cancer treatment.",
                    metadata={"source": "sample", "type": "market_analysis"}
                )
            ]
        
        print(f"Processing {len(all_docs)} documents...")
        
        # Split documents
        split_docs = []
        for doc in all_docs:
            splits = self.ingestion.text_splitter.split_documents([doc])
            split_docs.extend(splits)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=Config.VECTOR_STORE_PATH
        )
        
        print(f"Knowledge base built with {len(split_docs)} document chunks")
        
    def get_retriever(self, k: int = 5):
        """Get retriever for the knowledge base"""
        if not self.vector_store:
            raise ValueError("Knowledge base not built yet. Call build_knowledge_base() first.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

class PharmaRAGChains:
    """RAG chains for different aspects of pharma research"""
    
    def __init__(self, knowledge_base: PharmaKnowledgeBase):
        self.kb = knowledge_base
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0)
        
    def create_regulatory_chain(self):
        """Chain focused on regulatory requirements and FDA guidance"""
        template = """
        You are a pharmaceutical regulatory expert. Use the following context to answer questions about regulatory requirements, FDA guidance, and compliance considerations.
        
        Context: {context}
        
        Question: {question}
        
        Provide a detailed, accurate response focusing on:
        1. Regulatory requirements
        2. FDA guidance and recommendations
        3. Compliance considerations
        4. Timeline implications
        
        Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.kb.get_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
    
    def create_competitive_analysis_chain(self):
        """Chain for competitive landscape analysis"""
        template = """
        You are a pharmaceutical market analyst. Use the following context to provide competitive intelligence and market analysis.
        
        Context: {context}
        
        Question: {question}
        
        Provide analysis covering:
        1. Competitive landscape
        2. Market positioning opportunities
        3. Competitive advantages/disadvantages
        4. Strategic recommendations
        
        Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.kb.get_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
    
    def create_clinical_development_chain(self):
        """Chain for clinical development insights"""
        template = """
        You are a clinical development expert in pharmaceuticals. Use the following context to provide insights on clinical trials, development strategies, and timelines.
        
        Context: {context}
        
        Question: {question}
        
        Focus your response on:
        1. Clinical trial design considerations
        2. Development timeline estimates  
        3. Risk assessment
        4. Success probability factors
        
        Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.kb.get_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )

# Usage example
if __name__ == "__main__":
    # Initialize knowledge base
    kb = PharmaKnowledgeBase()
    kb.build_knowledge_base("oncology", "cancer")
    
    # Create RAG chains
    chains = PharmaRAGChains(kb)
    regulatory_chain = chains.create_regulatory_chain()
    
    # Test query
    response = regulatory_chain.run("What are the key FDA requirements for oncology drug approval?")
    print(response)