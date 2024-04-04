import docx
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
# from langchain_community.embeddings.edenai import EdenAiEmbeddings
from langchain_community.llms import EdenAI
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# os.environ['EDENAI_API_KEY'] = userdata.get('EDEN_KEY')

from getpass import getpass
import os

load_dotenv()
my_secret = os.getenv('GOOGLE_API_KEY')

# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass("AIzaSyBMCvkIaBkOqEo_dqhXzQQQ8jye4e0FM1U")
#     my_secret = os.environ.get('GOOGLE_API_KEY')


class DocumentProcessor:
    def __init__(self, path_arg):
        self.doc_path = path_arg

    # files reading functions
    def for_pdf(self, dir_path):
        with open(dir_path, 'rb') as pfile:
            pdf_r = PdfReader(pfile)
            text = ""
            for page in range(len(pdf_r.pages)):
                text+=pdf_r.pages[page].extract_text()
        return text

    def for_doc(self, dir_path):
        with open(dir_path, 'r'):
            doc_r = docx.Document(dir_path)
            text = ""
            for par in doc_r.paragraphs:
                text += par.text + "\n"
        return text

    def for_text(self, dir_path):
        with open(dir_path, 'r') as file:
            text = file.read()
        return text

    #general document reading function
    def readFilez(self):
        combined_txt = ""
        for filename in os.listdir(self.doc_path):
            file_path = os.path.join(self.doc_path, filename)
            if filename.endswith('.txt'):
                combined_txt += self.for_text(file_path)
            elif filename.endswith('.docx'):
                combined_txt += self.for_doc(file_path)
            elif filename.endswith('.pdf'):
                combined_txt += self.for_pdf(file_path)
        return combined_txt


class ChainProcessor:
    def __init__(self, doc_text):
        self.text = doc_text

    def CProcessing(self):
        # split text into chunks for easy processing and memory management
        char_txt_splitter = CharacterTextSplitter(
            separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)

        text_chunks = char_txt_splitter.split_text(self.text)

        #embeddings = EdenAiEmbeddings(provider='openai')
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        docsearch = FAISS.from_texts(text_chunks,embeddings)
        llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=my_secret)
        
        
        template = """You are a UniVIsor A guide for university and career paths, you provide professional guidance and answers peoples queries

            Given the following extracted parts of a long document and a question, create a final well formatted answer.

            {context}

            {chat_history}
            Human: {query}
            Chatbot:"""
        
        prompt = PromptTemplate(input_variables=["chat_history", "human_input", "context"], template=template)
        
        # memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
        memory = ConversationBufferWindowMemory(chat_memory="chat_history", k=3, input_prefix="query", output_prefix="response")
        
        chain = load_qa_chain(llm=llm, chain_type='stuff', memory=memory, prompt=prompt)

        return docsearch, chain

    def generate_response(self, query, doc_and_chain):
        docsearch, chain = doc_and_chain
        docs = docsearch.similarity_search(query)
        response = chain.run({"input_documents": docs, "query": query}, return_only_outputs=True)
        # response = chain.run(input_documents=docs, question=query)
        return response
