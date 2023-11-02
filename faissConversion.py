import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain

# Load text file
with open('./captions.txt', 'r') as file:
    text = file.read()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=0, length_function=len)
chunks = text_splitter.split_text(text)

# Create documents from chunks
docs = text_splitter.create_documents(chunks)

# Convert chunks to embeddings and save as FAISS file
embedding = GooglePalmEmbeddings(google_api_key="AIzaSyBysL_SjXQkJ8lI1WPTz4VwyH6fxHijGUE")
vdb_chunks_HF = FAISS.from_documents(docs, embedding=embedding)
vdb_chunks_HF.save_local("vdb_chunks_captions", index_name="indexCaptions")

# # Load language model
# llm = HuggingFaceHub(huggingfacehub_api_token="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", model_name="bert-large-uncased-whole-word-masking-finetuned-squad")

# # Load question answering chain
# chain = load_qa_chain(llm, chain_type="stuff")

# # Load FAISS file and perform similarity search
# vectorstore = FAISS.load_local("vdb_chunks_HF", embedding)
# qs = "What is Turing explaining about the imitation game?"
# docs2 = vectorstore.similarity_search(qs)

# # Run question answering chain
# response = chain.run(docs2[1].page_content, question=chain)
