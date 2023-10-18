
# ChatBot Research Tool: Enabling Informed Financial Insights

The News Research Tool encapsulates a rich, user-oriented platform crafted with the aim of streamlining the extraction and consumption of key insights, especially those dwelling within articles related to financial markets and varied economic zones.

![Snapshot_Streamlit_](https://github.com/EfthimiosVlahos/ChatBot_Research_Analysis/assets/56899588/f4b0ad93-24a2-417d-98d6-a7d4599be457)


## Core Features

- **URL Loading and Text Processing**: Enables users to either directly input URLs or upload text files to access article content, which is then processed using sophisticated NLP techniques.

- **Data Processing with LangChain's UnstructuredURLLoader**: Efficiently fetches and prepares textual content from articles ensuring it's in an optimal format for further analysis.

- **Embedding Vector Construction with OpenAI and FAISS**: Utilizes OpenAI's embedding capabilities and FAISS, a potent library facilitating similarity search, to streamline and enhance the retrieval of pertinent information.

- **Interactive Q&A with LLM (ChatGPT)**: Provides an interactive user interface where users can pose queries and receive accurate, concise answers, all backed with source URLs.



## Installation

1.Clone this repository to your local machine using:

```bash
  git clone https://github.com/codebasics/langchain.git
```
2.Navigate to the project directory:

```bash
  cd 2_news_research_tool_project
```
3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4.Set up your OpenAI API key by creating a .env file in the project root and adding your API

```bash
  OPENAI_API_KEY=your_api_key_here
```
## Usage/Examples

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- On the sidebar, you can input URLs directly.

- Initiate the data loading and processing by clicking "Process URLs."

- Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.

- The embeddings will be stored and indexed using FAISS, enhancing retrieval speed.

- The FAISS index will be saved in a local file path in pickle format for future use.
- One can now ask a question and get the answer based on those news articles
- In the tutorial, I used following news articles
  - https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html
  - https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html
  - https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html

## Project Structure

- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- faiss_store_openai.pkl: A pickle file to store the FAISS index.
- .env: Configuration file for storing your OpenAI API key.


