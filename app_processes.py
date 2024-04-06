import docx
import os
import getpass
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_history_aware_retriever, RetrievalQA
from langchain.agents import create_vectorstore_agent, AgentExecutor, Tool, initialize_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from PyPDF2 import PdfReader


load_dotenv()
my_secret = os.environ.get('GOOGLE_API_KEY')

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("AIzaSyBMCvkIaBkOqEo_dqhXzQQQ8jye4e0FM1U")
    my_secret = os.environ.get('GOOGLE_API_KEY')


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
        char_txt_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        text_chunks = char_txt_splitter.split_text(self.text)
        
        persist_dir = 'vdb' #creating a vector db to disk
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") #initialize embeddins
        vector_db = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
        retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={"k": 3})
        
        llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=my_secret)
        
        memory = ConversationBufferWindowMemory(memory_key ="chat_history",k=3)
        
        vector_i = VectorStoreInfo(
            name = 'vector_db',
            description = 'Find useful information from the vector only.',
            vectorstore = vector_db
        )

        toolkit = VectorStoreToolkit(
            llm = llm,
            vectorstore_info = vector_i
        )
        
        tools =[ create_retriever_tool(
            retriever = retriever,
            name = "vector_tool",
            description = "Use this tool to retrieve information from the vector db as context",
            document_separator = "\n"
        )]
        
        template = """Answer the questions as UniVisor, a University and Career paths guide,
        Using the given context, be creative and empathetic.
        {chat_history}
        Human: {input}
        AI: """
        
        prompt = PromptTemplate( input_variables=["input", "chat_history"], template = template)
        
        agent_store = create_vectorstore_agent(
            llm = llm,
            toolkit = toolkit,
            verbose = True,
            agent_executor_kwargs = {"memory":memory},
            kwargs = {"prompt":prompt}
        )
        
        return agent_store


    def generate_response(self, query, chains): 
        chain = chains

        response = chain.invoke({ "input": query})       
        return response['output']
