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
Ollama serve
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



