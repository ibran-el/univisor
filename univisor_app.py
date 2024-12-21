import telebot 
import os
import glob
import uuid
import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  
from langchain_core.vectorstores import InMemoryVectorStore 
from langchain_cohere import CohereEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate 
from langchain.memory import ConversationBufferMemory 
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_core.chat_history import BaseChatMessageHistory 
from langchain_core.runnables.history import RunnableWithMessageHistory 
from langchain_groq import ChatGroq 
from langchain.chains.combine_documents import create_stuff_documents_chain 
import gsheets_db 


# APP SETUPS
load_dotenv()

token = os.environ.get('BOT_TOKEN')
groq = os.environ.get('GROQ_API_KEY')
cohere = os.environ.get('COHERE_API_KEY')

# load reference files (remember to put the mechanism for loading multiple files from a choosen location)
# file_path = "doc/udsm.pdf"
# loader = PyPDFLoader(file_path)

# docs = loader.load()
file_path = glob.glob("doc/*.pdf")

docs = []
for file in file_path:
    loader = PyPDFLoader(file)
    docs += loader.load()

print(len(docs))


# set up the LLM
llm = ChatGroq(
      model="llama3-8b-8192",
      temperature = 0.5
      )

# creating a document retriever
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key = cohere
        )
)

retriever = vectorstore.as_retriever()


# SETTING UP CONVERSATIONAL MEMORY-------------------------------
# creating the conversational memory prompt for context provision
context_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if it is really needed and otherwise return it as is."
)
# adding user  promt to context and embedding conversational history
contextualized_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", context_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# creating hisstory aware retriever from the context
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualized_prompt
)

# implementing memory
memory = ConversationBufferMemory(
    memory_key="chat_history",  # This is the key where memory is stored in the conversation
    return_messages=True  # Ensures previous messages are returned with each query
)


# IMPLEMENTING THE ANSWER GENERATION-------------------------------
# defining the agent
system_prompt = (
    "your name is univisor"
    "You assist students, parents, and prospective applicants with college-related questions. "
    "Your primary purpose is to provide information on admissions requirements, course details, and other frequently asked questions.\n"
    "you are professional, friendly and empathetic.\n\n"
    "Keep responses concise and not more than 3 senteces"
    "For any unclear or ambiguous questions, provide a general response and politely encourage the user to reach out to an advisor. "
    "You must ask follow-up questions, remembering previous interactions in the same conversation to maintain a coherent multi-turn dialogue.\n\n"
    "Avoid these phrases\: \'according to information provided\', \'According to the provided information\'"
    "always ((structure)) and properly ((format your answers)), use bullets or numbered lists"
    "make answers as simple as possible and as short as possible"
    "{context}"
)

# creating the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# creating answer generation and RAG chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



#implementing session for conversational context
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# google sheets db data entry
def enter_data(chat_id, session_id, timestamp, tg_username, sender, message):
    client = gsheets_db.gspread.authorize(gsheets_db.creds)
    sheet = client.open("univisor logs").sheet1
    sheet.append_row([chat_id, session_id, timestamp, tg_username, sender, message])

# response function
def generate_session_token():
    return str(uuid.uuid4())

def get_reponse(q):
     session_token = generate_session_token()
     results = conversational_rag_chain.invoke(
        {
            "input": f"{q}"
        },
        config={
             "configurable": {
                  "session_id": session_token}},) # constructs a key in `store`.
    
     return results['answer']
    
print("done")


# BOT SETUPS
unibot = telebot.TeleBot(token)

@unibot.message_handler(commands=['start', 'help'])
def send_welcome(message):
	name = message.from_user.first_name
	unibot.reply_to(message, 
        f""" Hello there {name}! 👋 I'm your UDSM Admission Assistant. Feel free to ask me anything about admissions to the University of Dar es Salaam. I can help you with:
        * Understanding admission requirements
        * Finding suitable courses based on your    qualifications
        * Getting clear and easy-to-understand    explanations

    Just send me a message and I'll do my best to assist you! 😊
        """)
     
@unibot.message_handler(func=lambda message: True)
def reply_to_message(msg):
    question = msg.text
    enter_data(msg.chat.id, msg.chat.id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg.from_user.username, "user", question)

    answer = get_reponse(question)
    enter_data(msg.chat.id, msg.chat.id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg.from_user.username, "bot", answer)
    unibot.send_message(chat_id = msg.chat.id, text = answer, parse_mode='Markdown')

unibot.infinity_polling(timeout=30,)
