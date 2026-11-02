"""
Medication Manager Agent - Handles drug interactions and medication queries.
Built with LangGraph for agent orchestration.
"""
from typing import TypedDict, Annotated, List, Dict, Any
import sys
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
from functools import wraps
import time

# Load environment variables
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.rag.retriever import MedicationKnowledgeRetriever
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import FAISSVectorStore
from src.utils.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, VECTOR_DB_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def timing(logger):
    """A decorator to log the execution time of a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"PERFORMANCE: {func.__name__} took {elapsed_time:.4f} seconds to execute.")
            return result
        return wrapper
    return decorator

class AgentState(TypedDict):
    """State for the medication manager agent."""
    query: str
    patient_context: Dict[str, Any]
    retrieved_knowledge: List[Dict]
    response: str
    confidence: float
    warnings: List[str]

class MedicationManagerAgent:
    """Agent for managing medication queries and drug interactions."""
    
    def __init__(self, google_api_key: str = None):
        """Initialize the medication manager agent."""
        logger.info("Initializing Medication Manager Agent...")
        
        # Initialize LLM with Google Gemini
        google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY in .env or pass it explicitly.")
            
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )
        
        # Initialize RAG components
        embedding_generator = EmbeddingGenerator()
        vector_store = FAISSVectorStore(dimension=embedding_generator.dimension)
        self.retriever = MedicationKnowledgeRetriever(embedding_generator, vector_store)
        
        # Load knowledge base
        try:
            self.retriever.load_knowledge_base(VECTOR_DB_DIR)
            logger.info("Loaded existing knowledge base")
        except Exception as e:
            logger.warning(f"Could not load knowledge base: {e}")
            logger.info("Please run setup_rag.py first")
        
        # Define prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert medication management assistant. Your role is to:
1. Answer questions about medications, drug interactions, and side effects
2. Provide personalized medication guidance based on patient context
3. Alert patients to potential risks and interactions
4. Encourage medication adherence with practical tips

Always:
- Be clear, concise, and accurate
- Include relevant warnings and precautions
- Recommend consulting healthcare providers for medical decisions
- Use retrieved knowledge to support your answers

Context about the patient:
{patient_context}

Retrieved Medical Knowledge:
{retrieved_knowledge}
"""),
            ("human", "{query}")
        ])
        
        logger.info(" Medication Manager Agent initialized successfully")
    
    @timing(logger)
    def retrieve_knowledge(self, query: str, patient_context: Dict = None) -> List[Dict]:
        """
        Retrieve relevant medical knowledge for the query.
        
        Args:
            query: User query
            patient_context: Optional patient information
            
        Returns:
            List of retrieved documents
        """
        # Enhance query with patient context
        enhanced_query = query
        if patient_context and 'medications' in patient_context:
            meds = ", ".join(patient_context['medications'])
            enhanced_query = f"{query} (Patient medications: {meds})"
        
        logger.info(f"Retrieving knowledge for: {enhanced_query}")
        results = self.retriever.retrieve(enhanced_query, top_k=5)
        
        return results
    
    @timing(logger)
    def detect_warnings(self, query: str, patient_context: Dict, retrieved_knowledge: List[Dict]) -> List[str]:
        """
        Detect potential warnings based on query and patient context.
        
        Args:
            query: User query
            patient_context: Patient information
            retrieved_knowledge: Retrieved medical knowledge
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check for interaction keywords in retrieved knowledge
        interaction_keywords = ['interaction', 'risk', 'avoid', 'caution', 'warning']
        for doc in retrieved_knowledge:
            content_lower = doc['content'].lower()
            if any(keyword in content_lower for keyword in interaction_keywords):
                if doc['similarity'] > 0.7:  # High similarity = relevant warning
                    warnings.append(f" Important: {doc['content'][:200]}...")
        
        # Check patient medications for common interactions
        if patient_context and 'medications' in patient_context:
            meds = patient_context['medications']
            known_interactions = [
                (['Lisinopril', 'Ibuprofen'], 'ACE inhibitors and NSAIDs may interact'),
                (['Warfarin', 'Aspirin'], 'Increased bleeding risk'),
                (['Metformin', 'Alcohol'], 'Risk of lactic acidosis')
            ]
            
            for med_pair, warning in known_interactions:
                if all(med in meds for med in med_pair):
                    warnings.append(f" Drug Interaction Alert: {warning}")
        
        return warnings
    
    @timing(logger)
    def generate_response(self, state: AgentState) -> AgentState:
        """
        Generate response using LLM with RAG context.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with response
        """
        # Format patient context
        patient_info = "No patient context provided"
        if state.get('patient_context'):
            ctx = state['patient_context']
            patient_info = f"""
            - Medications: {', '.join(ctx.get('medications', []))}
            - Conditions: {', '.join(ctx.get('conditions', []))}
            - Allergies: {', '.join(ctx.get('allergies', []))}
            """
        
        # Format retrieved knowledge
        knowledge_text = "\n\n".join([
            f"[Source {i+1}] (Relevance: {doc['similarity']:.2f})\n{doc['content']}"
            for i, doc in enumerate(state.get('retrieved_knowledge', []))
        ])
        
        # Generate response
        messages = self.prompt_template.format_messages(
            query=state['query'],
            patient_context=patient_info,
            retrieved_knowledge=knowledge_text
        )
        
        response = self.llm.invoke(messages)
        
        # Update state
        state['response'] = response.content
        state['confidence'] = 0.85  # Placeholder - could be calculated based on retrieval scores
        
        logger.info("Generated response successfully")
        return state
    
    @timing(logger)
    def process_query(self, query: str, patient_context: Dict = None) -> Dict[str, Any]:
        """
        Process a medication query end-to-end.
        
        Args:
            query: User query
            patient_context: Optional patient information
            
        Returns:
            Response dictionary with answer and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # Initialize state
        state: AgentState = {
            'query': query,
            'patient_context': patient_context or {},
            'retrieved_knowledge': [],
            'response': '',
            'confidence': 0.0,
            'warnings': []
        }
        
        # Retrieve knowledge
        state['retrieved_knowledge'] = self.retrieve_knowledge(query, patient_context)
        
        # Detect warnings
        state['warnings'] = self.detect_warnings(query, patient_context or {}, state['retrieved_knowledge'])
        
        # Generate response
        state = self.generate_response(state)
        
        result = {
            'answer': state['response'],
            'confidence': state['confidence'],
            'warnings': state['warnings'],
            'sources': [
                {
                    'content': doc['content'][:200] + '...',
                    'similarity': doc['similarity'],
                    'category': doc['metadata'].get('category', 'Unknown')
                }
                for doc in state['retrieved_knowledge'][:3]
            ]
        }
        
        logger.info(" Query processed successfully")
        return result

# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = MedicationManagerAgent()
    
    # Example patient context
    patient_context = {
        'medications': ['Lisinopril', 'Metformin', 'Atorvastatin'],
        'conditions': ['Hypertension', 'Diabetes', 'High Cholesterol'],
        'allergies': []
    }
    
    # Example queries
    queries = [
        "Can I take Ibuprofen for my headache?",
        "What time should I take my Metformin?",
        "I'm experiencing muscle pain. Is this related to my medications?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = agent.process_query(query, patient_context)
        
        print(f"\nAnswer:\n{result['answer']}")
        
        if result['warnings']:
            print(f"\n Warnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
        
        print(f"\nConfidence: {result['confidence']:.2%}")
        print(f"\nSources: {len(result['sources'])} documents retrieved")
