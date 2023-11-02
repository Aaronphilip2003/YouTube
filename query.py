from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings

# Load language model
llm = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 64},
)

# Load question answering chain
chain = load_qa_chain(llm, chain_type="stuff")

# Load FAISS file
embedding = GooglePalmEmbeddings(google_api_key="AIzaSyBysL_SjXQkJ8lI1WPTz4VwyH6fxHijGUE")
vectorstore = FAISS.load_local("vdb_chunks_captions", embedding)

# Ask a question
question = input("Enter your question: ")

# Perform similarity search
docs = vectorstore.similarity_search(question)

# Run question answering chain
response = chain.run(docs[0].page_content, question=question)

# Print the response
print(response)
