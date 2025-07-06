# Multi_source_langchain_openai_agent


This project demonstrates a Langchain agent capable of answering questions based on information retrieved from multiple sources: Wikipedia, web pages (specifically the Langsmith documentation), and arXiv. It leverages OpenAI's `gpt-3.5-turbo-0125` model for LLM tasks, `OllamaEmbeddings` for embeddings, and FAISS for efficient vector storage and similarity search.

## Overview

This project showcases how to build a powerful and versatile Langchain agent that can draw upon a variety of data sources to provide informed and comprehensive answers.  It combines:

*   **Diverse Data Sources:**  Wikipedia, web pages, and arXiv.
*   **OpenAI LLM:**  `gpt-3.5-turbo-0125` for natural language processing and reasoning.
*   **Ollama Embeddings:** `tinyllama` model for generating vector embeddings from the text.
*   **FAISS Vector Database:**  For fast and scalable similarity search within the indexed data.
*   **Langchain Framework:** For orchestrating the data loading, embedding, storage, and agent execution.

## Features

*   **Multi-Source Data Retrieval:**  Retrieves relevant information from Wikipedia, web pages, and arXiv based on user queries.
*   **Context-Aware Question Answering:** Answers questions based on the combined knowledge from the retrieved sources.
*   **Vector Database Search:** Uses FAISS to efficiently find the most relevant documents based on semantic similarity.
*   **Customizable:**  The agent can be easily extended to include additional data sources or different LLMs/embedding models.

## Requirements

*   Python 3.8+
*   Poetry (recommended)
*   OpenAI API key
*   Ollama installed and running with the `tinyllama` model available

## Installation



1.  **Install dependencies using Poetry (Recommended):**

    ```bash
    poetry install
    ```

    Or, if you prefer pip:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up your environment variables:**

    Create a `.env` file in the project root with the following content:

    ```
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    ```

    Replace `"YOUR_OPENAI_API_KEY"` with your actual OpenAI API key.

## Usage

1.  **Run the main script:**

    ```bash
    poetry run python main.py
    ```

    Or using pip:

    ```bash
    python main.py
    ```

2.  **Enter your question when prompted:**

    The script will load data from the specified sources, create embeddings, build the FAISS index, and then use the Langchain agent to answer your question based on the retrieved information.

## Code Structure

*   `OPEMAI_agent.ipynb`: Main script to run the agent, sets up the data loading, embedding, vector database, and Langchain agent.

*   `requirements.txt`: Python dependencies (if using pip).

*   `.env`: Environment variables (API keys).

## Implementation Details

1.  **Data Loading:**
    *   **Wikipedia:**  Uses Langchain's `WikipediaLoader` to retrieve articles based on search queries.
    *   **Web Page (Langsmith):** Uses Langchain's `WebBaseLoader` to load the content of the Langsmith webpage.
    *   **ArXiv:**  Uses Langchain's `ArxivLoader` to retrieve papers based on search queries.

2.  **Text Splitting:**
    *   Uses Langchain's `RecursiveCharacterTextSplitter` to split the loaded documents into smaller chunks for embedding. This helps improve the accuracy of similarity search.

3.  **Embeddings:**
    *   `OllamaEmbeddings(model="tinyllama")` is used to create vector embeddings of the text chunks.  Ollama must be installed and the `tinyllama` model available to it.

4.  **Vector Database (FAISS):**
    *   FAISS (Facebook AI Similarity Search) is used to store the embeddings and perform efficient similarity search.  The `FAISS.from_documents` method is used to create the index directly from the loaded documents and embeddings.

5.  **Langchain Agent:**
    *   A Langchain agent is created using `OpenAI` and the `gpt-3.5-turbo-0125` model.  This agent is configured to use the FAISS vector database as a tool for retrieving relevant information.
    *   The agent takes a user's question as input, retrieves relevant documents from the vector database, and then uses the LLM to generate an answer based on the retrieved information.
