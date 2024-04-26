## Project Description
This code implements a customer service assistant using Langchain, Ollama(llama2) locally. 
It takes a customer question and retrieves relevant data (likely financial transactions) to answer the question in a single, human-readable sentence.

We will be using a local, open source LLM “Llama2” through Ollama as then we don’t have to setup API keys and it’s completely free. 
However, you will have to make sure your device will have the necessary specifications to be able to run the model. 
For smallest models of 7B parameters, you need to have at least 8GB of RAM.

##### Installing Ollama
You can install Ollama using the guide provided on Ollama github: [Ollama guide](https://github.com/ollama/ollama?tab=readme-ov-file).

I am using a windows computer and I followed from the windows preview: [Windows Preview](https://github.com/ollama/ollama?tab=readme-ov-file#windows-preview)

Then run the Ollama setup file downloaded.

Once installed run this command to start the Ollama server:
```
ollama serve
```

##### Running the llama2 model locally using Ollama
After Installing Ollama, you pull the llama2 model which we'll be using in our project.
```
ollama pull llama2
```
I ran the model in shell instead to interact with the model before I started building retrieval chains with is on langchain.
You can run this command to directly chat with llama2. This command automatically pulls the model if it has not yet been pulled already:
```
ollama run llama2
```
You will be able to chat with the model directly through the terminal with this.

#### Running the programs in the notebooks
You should create a python virtual environment for the notebook.

- Install the dependencies:
```
pip install -r requirements.txt
```

- Run the cells in the `notebooks/chroma_ollama.ipynb` notebook one after the other.
- You can also run the python file `moneylion_qa_chain.py` in a python shell.


#### Code Guide

- Necessary imports
```
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import CSVLoader, PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain, RetrievalQA

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

```

- Define the LLM and the embedding model. We're using llama2 by Meta.
  
   `llm = Ollama(model="llama2")`
 
   `embeddings = OllamaEmbeddings(model="llama2")`

- Load the Documents using Langchain document loaders.
  
  `dictionary = "../data/Data Dictionary.pdf"`
  
  `data = "../data/data.csv"`

  `dict_loader = PyPDFLoader(dictionary)`
  
  `data_loader = CSVLoader(data)`

  `data_docs = data_loader.load()`
  
  `dict_doc = dict_loader.load()`

- Create or load the Vector Database from the data document using Chroma.
  
   Only a chunk of the data was stored in the Vector DB since it takes a really long time to create.

  A csv file with more than 20k lines will take hours.
  
  `data_chroma_db = Chroma.from_documents(data_docs[:100], embeddings, persist_directory="../data/100_chunk_chroma_db")`

This line creates the vector database.

But since I already created the database, I can just load it from where it was saved.

`data_chroma_db = Chroma(persist_directory="../data/100_chunk_chroma_db", embedding_function=embeddings)`

- Create a vector retriever from the Vector DB
  
  `data_retriever = data_chroma_db.as_retriever()`

- The Prompt Template
```
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
```


- The chain
```
from operator import itemgetter

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
  
```

- Run the chain.
```
reply = invoke_chain(question="How much did I spend on August 1st, 2023?", client_id=86)

```

