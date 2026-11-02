
# Agentic AI Medication Management System: Project Documentation

**Version:** 1.0.0
**Date:** 2025-10-19

---

## 1. Executive Summary & Conference Pitch

**The Pitch:** "In today's complex healthcare landscape, medication non-adherence costs billions and compromises patient outcomes. We present an Agentic AI Medication Management System, a multi-agent platform designed to provide personalized, real-time medication support. Our system combines three specialized AI agents—a Monitor, a Medication Manager, and a Nurse—to create a holistic ecosystem that not only tracks adherence but actively coaches patients, answers complex medical questions, and flags critical safety issues like drug interactions and allergies. By leveraging Retrieval-Augmented Generation (RAG) with a FAISS vector database and the power of Google's Gemini 2.5 Flash model, our system delivers empathetic, context-aware, and clinically-grounded support, demonstrating a scalable solution to one of healthcare's most persistent challenges."

**Problem:** Medication non-adherence is a major public health issue, leading to poor health outcomes and increased healthcare costs. Patients are often overwhelmed, lack understanding of their medications, or forget to take them.

**Solution:** A multi-agent AI system that acts as a personal medication assistant. It provides:
- **Intelligent Monitoring:** Proactive tracking of adherence patterns.
- **Expert Knowledge:** Instant, accurate answers to medication questions.
- **Empathetic Coaching:** Personalized reminders and motivational support.
- **Safety First:** Automated drug interaction and allergy checks.

---

## 2. System Architecture: A Multi-Agent Approach

The system is built on a **multi-agent architecture**, where each AI agent has a specialized role. This modular design allows for separation of concerns and makes the system robust and scalable.

- **Orchestrator (`main.py`):** The central hub that directs tasks to the appropriate agent based on user commands.
- **Monitor Agent (`monitor_agent.py`):** The data analyst. It works without an LLM and focuses on quantitative data: checking for drug interactions from a known database and analyzing adherence statistics.
- **Medication Manager Agent (`medication_manager.py`):** The pharmacist. This agent uses **Retrieval-Augmented Generation (RAG)** to answer complex questions. It retrieves relevant medical documents from a vector database before using the LLM to synthesize an answer, ensuring responses are grounded in factual knowledge.
- **Nurse Agent (`nurse_agent.py`):** The caregiver. This agent uses the LLM to generate empathetic, personalized coaching messages, reminders, and educational content based on the patient's adherence patterns and profile.

![System Architecture](figures/system_architecture.png)

---

## 3. Core Technologies, Methods, and Libraries

This project integrates state-of-the-art technologies to deliver its capabilities.

### Core Technologies
| Technology | Purpose |
| :--- | :--- |
| **Python 3.13** | The core programming language for the entire system. |
| **Google Gemini 2.5 Flash** | The Large Language Model (LLM) used for generating human-like text for the Nurse and Medication Manager agents. Chosen for its speed and advanced reasoning. |
| **LangChain** | The primary framework for building applications with LLMs. Used for prompt templating, managing agent state, and integrating with the LLM. |
| **FAISS (Facebook AI Similarity Search)** | A high-performance vector database used to store and search embeddings of medical documents for the RAG system. |
| **Sentence-Transformers** | A library used to generate high-quality embeddings (vector representations) of text documents. We use the `all-MiniLM-L6-v2` model. |

### Key Methodologies
| Method | Description |
| :--- | :--- |
| **Multi-Agent System** | A decentralized architecture where multiple autonomous agents collaborate to solve a problem. Our system uses three agents with distinct roles. |
| **Retrieval-Augmented Generation (RAG)** | A technique to make LLM responses more factual and up-to-date. The Medication Manager Agent first *retrieves* relevant medical documents from our FAISS database and then uses that context to *generate* an informed answer. |
| **Vector Search** | The core of our RAG system. Medical documents are converted into numerical vectors (embeddings). When a user asks a question, their query is also converted to a vector, and FAISS finds the most similar document vectors. |
| **Synthetic Data Generation** | We used the `Faker` library to generate a realistic dataset of 100 patients and 15,000+ medication records, allowing for robust testing without compromising real patient privacy. |

### Python Libraries
| Library | Usage |
| :--- | :--- |
| `langchain-google-genai` | Integrates Google's Gemini models with the LangChain framework. |
| `faiss-cpu` | The CPU version of the FAISS library for vector storage and search. |
| `sentence-transformers` | Generates text embeddings for the RAG system. |
| `pandas` & `numpy` | Essential for data manipulation, especially for handling patient and medication schedule data. |
| `dotenv` | Manages environment variables, keeping API keys and configurations secure. |
| `argparse` | Parses command-line arguments for running the system in different modes. |

---

## 4. Codebase Explained: File-by-File

This section provides a detailed explanation of every Python file in the project.

### 4.1. Main Application Files

#### `main.py`
- **Purpose:** The main entry point and orchestrator of the application. It parses user commands and delegates tasks to the appropriate agent.
- **Key Functions:**
    - `MedicationManagementSystem`: The main class that initializes and holds instances of all three agents.
    - `comprehensive_patient_check()`: Runs a full analysis for a patient, involving all three agents.
    - `interactive_mode()`: Provides a command-line interface (CLI) for interacting with the system.
    - `main()`: Parses command-line arguments (`--mode`, `--patient`) to determine which function to run.

#### `run_full_demo.py`
- **Purpose:** A script designed to run a complete, automated demonstration of the system's capabilities across multiple patients.
- **Functionality:** It iterates through a predefined list of patients (`P0001`, `P0002`, `P0003`) and runs a full analysis for each, showcasing every agent's functionality in a continuous flow. This is ideal for conference presentations.

#### `quick_demo.py`
- **Purpose:** A simplified, faster demonstration script that runs a single patient check using a subprocess.
- **Functionality:** It calls `main.py` for one patient and then prints a summary of the system's features and statistics.

#### `test_system.py`
- **Purpose:** A system verification script to ensure all components are set up correctly before running the full application.
- **Functionality:** It checks for the existence of patient data, verifies the RAG vector database, and tests the initialization of the Monitor Agent (which requires no API key). It provides a quick "health check" of the system.

---

### 4.2. The AI Agents (`src/agents/`)

#### `medication_manager.py`
- **Purpose:** The "pharmacist" agent. It answers complex medical questions using a RAG pipeline.
- **Core Logic:**
    1.  Initializes `ChatGoogleGenerativeAI` as the LLM.
    2.  Initializes the RAG components: `EmbeddingGenerator`, `FAISSVectorStore`, and `MedicationKnowledgeRetriever`.
    3.  `process_query()`: Takes a user's question and patient context.
    4.  `retrieve_knowledge()`: The retriever searches the FAISS database for relevant medical documents.
    5.  `generate_response()`: The agent constructs a detailed prompt containing the original question, patient context, and the retrieved documents. It sends this to the Gemini model to generate a factually-grounded answer.

#### `nurse_agent.py`
- **Purpose:** The "caregiver" agent. It provides empathetic coaching and reminders.
- **Core Logic:**
    1.  Initializes `ChatGoogleGenerativeAI` with a higher `temperature` (0.8) to encourage more creative and empathetic responses.
    2.  `provide_adherence_coaching()`: Takes patient info and adherence data. It uses a detailed system prompt to instruct the LLM to act as a compassionate nurse, analyze the adherence pattern, and provide practical, encouraging advice.
    3.  `generate_reminder()`: Creates simple, friendly medication reminders.

#### `monitor_agent.py`
- **Purpose:** The "data analyst" agent. It performs rule-based checks and statistical analysis. **It does not use an LLM.**
- **Core Logic:**
    1.  `check_medication_interactions()`: Checks a patient's medication list against a predefined dictionary of known drug interactions.
    2.  `generate_monitoring_report()`: Analyzes a patient's medication schedule (a pandas DataFrame) to calculate adherence rates, identify trends (e.g., declining adherence), and flag risks based on predefined thresholds.

---

### 4.3. Retrieval-Augmented Generation (`src/rag/`)

#### `embeddings.py`
- **Purpose:** Handles the creation of text embeddings.
- **Core Logic:**
    - `EmbeddingGenerator`: A class that loads the `sentence-transformers/all-MiniLM-L6-v2` model.
    - `embed_documents()`: Takes a list of text documents and converts each one into a 384-dimensional numerical vector.

#### `vector_store.py`
- **Purpose:** Manages the FAISS vector database.
- **Core Logic:**
    - `FAISSVectorStore`: A class that wraps the `faiss` library.
    - `add_documents()`: Adds document embeddings to the FAISS index.
    - `search()`: Takes a query vector and returns the most similar document vectors from the index.
    - `save_local()` / `load_local()`: Persists the FAISS index and document metadata to disk.

#### `retriever.py`
- **Purpose:** The bridge between the agent and the vector store.
- **Core Logic:**
    - `MedicationKnowledgeRetriever`:
    - `retrieve()`: Takes a text query, generates an embedding for it, and uses the `vector_store` to find the most relevant documents. It also filters the results based on a similarity score threshold (e.g., 0.7) to ensure only highly relevant documents are returned.

---

### 4.4. Utilities & Scripts (`src/utils/` & `scripts/`)

#### `src/utils/config.py`
- **Purpose:** Centralizes all configuration variables.
- **Functionality:** Loads API keys and other settings from the `.env` file using `python-dotenv`. This keeps sensitive information out of the source code.

#### `src/utils/logger.py`
- **Purpose:** Sets up a standardized logging format for the entire application.
- **Functionality:** Configures Python's `logging` module to produce clean, timestamped logs with severity levels (INFO, ERROR, etc.), which is crucial for debugging.

#### `scripts/generate_data.py`
- **Purpose:** A one-time script to create the synthetic dataset.
- **Functionality:** Uses the `Faker` library to generate 100 realistic but fictional patients, including names, conditions, and medications. It also creates a detailed medication schedule CSV file with randomized adherence data.

#### `scripts/setup_rag.py`
- **Purpose:** A one-time script to build the knowledge base for the RAG system.
- **Functionality:**
    1.  Reads plain-text medical documents from the `data/medical_guidelines/` directory.
    2.  Uses the `EmbeddingGenerator` to convert these documents into vectors.
    3.  Saves the vectors into a FAISS index (`index.faiss`) and saves the corresponding document text (`documents.pkl`).

---

## 5. How to Present This Project

Here is a suggested flow for a conference presentation:

1.  **Start with the Pitch (Section 1):** Grab the audience's attention by highlighting the real-world problem and your innovative AI solution.
2.  **Explain the Architecture (Section 2):** Use the architecture diagram to visually explain the multi-agent system. Describe the role of each agent, emphasizing the "separation of concerns" (analyst, pharmacist, caregiver).
3.  **Dive into the "Magic" - RAG (Section 3):** Explain what RAG is and why it's crucial. Show how it prevents the LLM from making up facts. Use a diagram to illustrate the flow: `Query -> Embed -> Search -> Retrieve -> Augment -> Generate`.
4.  **Showcase the Code (Section 4):** Briefly walk through the key files.
    - Show `main.py` as the orchestrator.
    - Contrast the three agents: `monitor_agent.py` (pure Python logic), `nurse_agent.py` (creative LLM), and `medication_manager.py` (factual LLM + RAG).
    - Mention the `scripts` to show how the project's data foundation was built.
5.  **Live Demo (or Video):** Run `run_full_demo.py`. Pause at each step to explain what's happening:
    - "First, the Monitor Agent runs a quick, offline safety check."
    - "Next, the Medication Manager is using RAG to consult its knowledge base..."
    - "Finally, the Nurse Agent is generating a personalized coaching message based on the patient's 67% adherence rate."
6.  **Conclude with Impact:** Reiterate the benefits: improved patient safety, better adherence, and scalable personalized care. Discuss future work, such as integrating real-time data or adding more specialized agents.
