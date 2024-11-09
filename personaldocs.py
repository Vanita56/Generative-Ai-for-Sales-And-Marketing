from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from PyPDF2 import PdfReader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import magic
import os
import nltk

# openai_api_key = os.getenv("OPENAI_API_KEY", "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")

openai_api_key = os.getenv("OPENAI_API_KEY", "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")

loader = DirectoryLoader("C:\\Users\\Admin\\Desktop\\MPRsalesnmarketing", glob='**/*.txt')

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

docsearch = FAISS.from_documents(texts, embeddings)

llm = OpenAI(openai_api_key=openai_api_key)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

query = "How RapidRoad uses data analytics?"
qa.run(query)
print(qa.run(query))