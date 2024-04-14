import os

chroma_folder = os.listdir("data/100_chunk_chroma_db")

chroma_exists = False
for content in chroma_folder:
    if os.path.isdir("data/100_chunk_chroma_db"+f"/{content}"):
        chroma_exists=True
        print("Chroma db exists:", chroma_exists)
        
if chroma_exists:
    from langchain.llms import Ollama
    from langchain.embeddings import OllamaEmbeddings
    from langchain.document_loaders import CSVLoader, PyPDFLoader
    from langchain_community.vectorstores import Chroma

    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    from operator import itemgetter


    llm = Ollama(model="llama2")
    embeddings = OllamaEmbeddings(model="llama2")

    dict_chroma_db = Chroma(persist_directory="data/dict_chroma_db", embedding_function=embeddings)
    data_chroma_db = Chroma(persist_directory="data/100_chunk_chroma_db", embedding_function=embeddings)

    data_retriever = data_chroma_db.as_retriever()

    template="""
            You are a helpful assistant that serves as customer care service for our company.
            You asnwer their questions.
            If you don't have their answer, reply that you cannot help in the moment.
                    
            You have this data about the customers, provided by the company: {context}
            
            please answer this customer's question directly to them: {question}
            Their client clnt_id: {client_id}
            
            
            Make sure you answer in one human readable sentence.
            Answer directly, like you are in charge.
            Don't add extra informations or extra questions, don't refer to your sources and don't let the customers know about the context documents in your output.
            
            This is what each column/keyword means:
            Column ->> Description
            clnt_id ->> Client ID
            bank_id ->>  Bank ID
            acc_id ->>  Account ID
            txn_id ->> Transaction ID
            txn_date ->> Transaction date
            desc ->> Description
            amt ->> Amount
            cat ->> Category of the transaction
            merchant ->> Merchant of the transaction
            """
            
    prompt = ChatPromptTemplate.from_template(template)

    def invoke_chain(question: str, client_id=6):
        chain = (
            {
                'context': itemgetter("question") | data_retriever, "question": itemgetter("question"), "client_id": itemgetter('client_id'), 
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain.invoke({
            "client_id": client_id,
            "question": question
        })
        
        
    reply = invoke_chain(question="How much did I spend on August 1st, 2023?", client_id=86)
    print(reply)
    
else:
    raise Exception("Chroma db is required to create retriever.")