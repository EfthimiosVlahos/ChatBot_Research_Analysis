# End-to-End NLP Research Tool Project 

A sophisticated NLP research tool, facilitating the intelligent querying of a multitude of news articles by transforming them into a structured and searchable knowledge base using LangChain, OpenAI, and Streamlit.

![Snapshot_Streamlit_](https://github.com/EfthimiosVlahos/ChatBot_Research_Analysis/assets/56899588/ed892a39-abbf-4642-aa9c-dfad66c1d842)


## Table of Contents
- [URL Loading and Text Processing](#url-loading-and-text-processing)
- [Data Processing with LangChain's UnstructuredURLLoader](#data-processing-with-langchains-unstructuredurlloader)
- [Embedding Vector Construction with OpenAI and FAISS](#embedding-vector-construction-with-openai-and-faiss)
- [Usage](#usage)

## URL Loading and Text Processing <a id="url-loading-and-text-processing"></a>

The journey of my end-to-end NLP Research Tool begins with the adept loading of URLs and thorough text processing. This foundational phase is critical, ensuring that subsequent components of my project, such as data processing and vector embedding, operate with optimally formatted data.

### URL Loading

My tool is designed to accommodate user flexibility by accepting information in multiple formats:
- **Direct URL Input**: Users can directly input a list of URLs, from which the tool fetches textual content.
- **Text File Upload**: Alternatively, users can upload a text file containing a list of URLs, ensuring ease of use and the capability to manage multiple sources with seamlessness.

I've implemented a robust URL loader that not only retrieves article content but also ensures the extraction of relevant textual data, omitting any superfluous elements like advertisements or unrelated links.

### Text Processing

Once the textual data is secured, it is subjected to a series of NLP techniques to enhance its usability in subsequent stages. My text processing encompasses:
- **Tokenization**: Decomposing the text into smaller pieces, often words, to facilitate more nuanced analysis.
- **Stop Word Removal**: Eradicating common words (such as "is", "the", etc.) that do not contribute to understanding context or conducting searches.
- **Lemmatization**: Reducing words to their base or root form (e.g., transforming "running" to "run"), ensuring consistency and simplifying complexity.
- **Named Entity Recognition (NER)**: Identifying and classifying named entities in the text, such as person names, organizations, or locations.
- **Semantic Analysis**: Applying techniques to comprehend the meaning and context behind the words in the text.

This meticulous text processing ensures that the extracted information is in a coherent and streamlined format, ready for subsequent embedding and interactive Q&A stages. By employing these systematic and comprehensive text processing strategies, my tool ensures that user queries are addressed with accuracy and contextual relevance throughout the application.

_Note: Future iterations of the tool will continue to refine and expand upon these processes, enhancing accuracy, and usability based on user feedback and technological advancements._


## Data Processing with LangChain's UnstructuredURLLoader <a id="data-processing-with-langchains-unstructuredurlloader"></a>

In the journey of constructing the NLP Research Tool, an indispensable element was harnessing the functionalities of LangChain's UnstructuredURLLoader, crucial for adeptly managing the data processing stage, especially in efficiently retrieving and preparing textual content from various articles.

### UnstructuredURLLoader in Action

The **UnstructuredURLLoader** from LangChain has been instrumental in acquiring textual data from the provided URLs, ensuring it's formatted appropriately for analysis:

- **Text Retrieval**: For a given URL, the loader extracts all the text available on the webpage. It meticulously navigates through HTML and CSS, ensuring that the extracted text is free from code or unwanted characters.

- **Data Cleanup**: Once the raw text data is fetched, it undergoes an initial phase of processing. This includes removing unnecessary whitespaces, decoding characters, and handling any inconsistencies in encoding formats to yield a clean, consistent text.

- **Managing Various Textual Content Types**: The UnstructuredURLLoader is versatile, capable of managing various formats of textual content, whether they be articles, blogs, or forums. This ensures the tool is adaptable and useful across different data sources.

### Integration into the Project

Within the framework of my project, this phase of data retrieval and preparation lays the foundation for the subsequent phases of analysis and interaction. Here’s a brief look at the implementation:

```python
from langchain.document_loaders import UnstructuredURLLoader

# Instantiating the loader
loader = UnstructuredURLLoader()

# Loading the data from a sample URL
data = loader.load("https://example.com/sample-article")
```


## Embedding Vector Construction with OpenAI and FAISS <a id="embedding-vector-construction-with-openai-and-faiss"></a>

In my venture to elevate the efficacy of information retrieval in the project, I utilized OpenAI's potent embedding capabilities and integrated them with Facebook AI's Similarity Search (FAISS) to construct and manage embedding vectors.

### Constructing Embeddings with OpenAI

The essentiality of embedding arises from its numerical representation of textual chunks, which subsequently plays a pivotal role in semantic search — not through mere keyword matching, but through comprehending the intent and context of a user query. Here's an illustrative snippet:

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-mpnet-base-v2")
vectors = encoder.encode(df.text)
```

## Usage <a id="usage"></a>

In crafting this NLP Research Tool, my vision has been to simplify the research journey, making it accessible and resourceful for both beginners and seasoned researchers alike. 

### How to Utilize the Tool for Research Purposes

Embark on your research adventure with the NLP Research Tool by following these straightforward steps:

1. **Input Your URLs or Text Files**:
   Navigate to the URL input section. Here, you can either:
   - Directly input URLs into the designated field.
   - Upload a text file that contains a list of URLs.

2. **Initiate Text Extraction and Processing**:
   Once URLs are provided, the tool diligently extracts textual content, omitting unnecessary information like advertisements and ensures meticulous processing through techniques like tokenization, lemmatization, and Named Entity Recognition (NER).

3. **Commence Data Analysis**:
   Post text processing, the tool progresses towards data analysis, utilizing various Natural Language Processing (NLP) techniques, and creates a base for detailed exploration and findings. 

4. **Leverage Advanced Options for In-depth Analysis**:
   Explore further by utilizing advanced features, like embedding vector construction and similarity searches, to delve deeper into textual content and uncover nuanced insights.

5. **Interactive Q&A Section**:
   Engage with the interactive Q&A section by inputting queries related to the extracted text. My tool not only fetches relevant information but also ensures accuracy and contextual relevance.

6. **Explore Generated Outputs**:
   Review and explore the generated outputs, ensuring the retrieved and processed data aligns with your research objectives.


