"""
Setup RAG pipeline with medical knowledge base.
"""
import argparse
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent))
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import FAISSVectorStore
from src.rag.retriever import MedicationKnowledgeRetriever
from src.utils.config import MEDICAL_GUIDELINES_DIR, VECTOR_DB_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "setup_rag.log")

# Medical knowledge base - Common medication guidelines
MEDICATION_KNOWLEDGE_BASE = [
    # Drug Interactions
    "Lisinopril and NSAIDs (like Ibuprofen): NSAIDs can reduce the effectiveness of ACE inhibitors like Lisinopril. They may also increase the risk of kidney problems when taken together. Monitor blood pressure and kidney function regularly.",
    
    "Metformin and Alcohol: Drinking alcohol while taking Metformin can increase the risk of lactic acidosis, a serious condition. Limit alcohol consumption and watch for symptoms like muscle pain, difficulty breathing, or unusual fatigue.",
    
    "Sertraline and Alprazolam: Combining these medications can increase sedation and respiratory depression. Use together only under close medical supervision. Be cautious when driving or operating machinery.",
    
    "Atorvastatin and Grapefruit: Grapefruit juice can significantly increase Atorvastatin levels in the blood, raising the risk of muscle damage and liver problems. Avoid grapefruit products while on this medication.",
    
    "Warfarin and Aspirin: Both medications increase bleeding risk. When used together, the risk is significantly higher. Regular INR monitoring is essential. Watch for signs of bleeding.",
    
    # Medication Adherence Guidelines
    "Medication Adherence Best Practices: Take medications at the same time each day. Use pill organizers, set phone reminders, and keep medications in visible locations. Missing doses can lead to treatment failure.",
    
    "Morning Medication Routine: Blood pressure medications like Lisinopril, Amlodipine, and Losartan are typically most effective when taken in the morning. This aligns with the body's natural blood pressure patterns.",
    
    "Diabetes Medication Timing: Metformin should be taken with meals to reduce gastrointestinal side effects. Insulin timing depends on the type (rapid-acting, long-acting). Always coordinate with meal schedules.",
    
    "Statin Administration: Statins like Atorvastatin and Simvastatin are often taken at bedtime because cholesterol production is highest during sleep. Follow your prescription instructions.",
    
    # Side Effects Management
    "Common Side Effects of ACE Inhibitors: Dry cough is a common side effect of medications like Lisinopril. Dizziness may occur, especially when standing up quickly. Report persistent cough to your doctor.",
    
    "Metformin Side Effects: Nausea, diarrhea, and stomach upset are common when starting Metformin. These usually improve over time. Take with food to minimize gastrointestinal effects.",
    
    "SSRI Side Effects: Sertraline and other SSRIs may cause nausea, sleep changes, and sexual side effects. Most side effects improve after 2-4 weeks. Do not stop abruptly without medical guidance.",
    
    "Statin Muscle Pain: Report unexplained muscle pain, tenderness, or weakness while taking statins. This could indicate rhabdomyolysis, a serious but rare side effect requiring immediate attention.",
    
    # Medication Safety
    "Medication Storage: Store medications in a cool, dry place away from direct sunlight. Bathrooms are not ideal due to humidity. Keep medications in original containers with labels intact.",
    
    "Expired Medications: Do not use medications past their expiration date. Effectiveness may be reduced, and some medications can become harmful. Safely dispose of expired medications at pharmacy take-back programs.",
    
    "Missed Dose Protocol: If you miss a dose, take it as soon as you remember unless it's almost time for the next dose. Never double up on doses. Contact your pharmacist if unsure.",
    
    # Specific Medication Guidelines
    "Lisinopril (ACE Inhibitor): Used for high blood pressure and heart failure. Monitor blood pressure regularly. Can cause elevated potassium levels. Avoid potassium supplements unless directed. May cause dizziness initially.",
    
    "Amlodipine (Calcium Channel Blocker): Used for hypertension and angina. May cause ankle swelling. Take at the same time daily. Grapefruit juice may increase levels. Report rapid heartbeat or severe dizziness.",
    
    "Metformin (Diabetes Medication): First-line treatment for type 2 diabetes. Take with meals. Can cause vitamin B12 deficiency with long-term use. Lactic acidosis is rare but serious - seek help if you have severe muscle pain.",
    
    "Atorvastatin (Statin): Lowers cholesterol to prevent heart disease and stroke. Avoid grapefruit. Report muscle pain. Regular liver function tests may be needed. Most effective when taken at bedtime.",
    
    "Sertraline (SSRI Antidepressant): Used for depression and anxiety disorders. May take 4-6 weeks for full effect. Do not stop suddenly. May cause initial nausea. Take with food if stomach upset occurs.",
    
    "Albuterol (Bronchodilator): Quick-relief inhaler for asthma. Use as needed for shortness of breath. Rinse mouth after use. If using more than twice weekly, asthma may not be well controlled - contact doctor.",
    
    "Omeprazole (Proton Pump Inhibitor): Reduces stomach acid for GERD and ulcers. Take 30-60 minutes before breakfast. Long-term use may affect nutrient absorption. Not for immediate heartburn relief.",
    
    # Patient Education
    "Why Medication Adherence Matters: Missing doses can lead to treatment failure, disease progression, and increased healthcare costs. For chronic conditions, consistent medication use is essential for managing symptoms and preventing complications.",
    
    "Communicating with Healthcare Providers: Always inform doctors and pharmacists about all medications, supplements, and allergies. Keep an updated medication list. Ask questions about new prescriptions. Report side effects promptly.",
    
    "Understanding Your Medications: Know the name, purpose, dosage, and timing of each medication. Understand potential side effects and what to watch for. Know which medications are essential and which are as-needed.",
    
    # Stress and Medication Adherence
    "Impact of Stress on Medication Adherence: High stress levels are associated with 40% lower medication adherence rates. Stress can cause forgetfulness and reduced motivation. Managing stress through relaxation techniques can improve adherence.",
    
    "Anxiety and Medication Taking: Anxiety about side effects or medication dependence can reduce adherence. Discuss concerns openly with healthcare providers. Understanding benefits and risks can reduce anxiety.",
    
    # Special Populations
    "Elderly Medication Management: Older adults often take multiple medications (polypharmacy). Use pill organizers, medication lists, and regular medication reviews with doctors. Simplify regimens when possible.",
    
    "Medication Adherence Barriers: Common barriers include cost, side effects, complex regimens, forgetfulness, and lack of symptoms. Identifying personal barriers is the first step to improving adherence.",
]

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def setup_medical_knowledge_base():
    """Setup the medical knowledge base for RAG."""
    logger.info("Setting up medical knowledge base...")
    
    # Create metadata for each document
    metadata_list = []
    for i, doc in enumerate(MEDICATION_KNOWLEDGE_BASE):
        # Extract category from first line
        category = "General"
        if "Interaction" in doc or "and" in doc.split(':')[0]:
            category = "Drug Interactions"
        elif "Adherence" in doc:
            category = "Adherence"
        elif "Side Effect" in doc:
            category = "Side Effects"
        elif "Safety" in doc or "Storage" in doc:
            category = "Safety"
        elif any(med in doc for med in ["Lisinopril", "Metformin", "Atorvastatin", "Sertraline", "Albuterol", "Omeprazole"]):
            category = "Medication Specific"
        elif "Stress" in doc or "Anxiety" in doc:
            category = "Behavioral"
        
        metadata_list.append({
            'id': i,
            'category': category,
            'source': 'Clinical Guidelines'
        })
    
    # Initialize components
    logger.info("Initializing RAG components...")
    embedding_generator = EmbeddingGenerator()
    vector_store = FAISSVectorStore(dimension=embedding_generator.dimension)
    retriever = MedicationKnowledgeRetriever(embedding_generator, vector_store)
    
    # Add knowledge base
    logger.info(f"Adding {len(MEDICATION_KNOWLEDGE_BASE)} documents to knowledge base...")
    retriever.add_knowledge_base(MEDICATION_KNOWLEDGE_BASE, metadata_list)
    
    # Save to disk
    logger.info("Saving knowledge base...")
    retriever.save_knowledge_base(VECTOR_DB_DIR)
    
    logger.info("RAG setup completed successfully!")
    logger.info(f"   - Total documents: {len(MEDICATION_KNOWLEDGE_BASE)}")
    logger.info(f"   - Vector DB saved to: {VECTOR_DB_DIR}")
    
    # Test retrieval
    logger.info("\n" + "="*50)
    logger.info("Testing retrieval with sample queries...")
    logger.info("="*50)
    
    test_queries = [
        "What are the interactions between Lisinopril and Ibuprofen?",
        "How should I take Metformin?",
        "What are the side effects of statins?"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=2)
        for i, result in enumerate(results, 1):
            logger.info(f"  Result {i} (similarity: {result['similarity']:.3f}):")
            logger.info(f"    {result['content'][:150]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup RAG pipeline for medication knowledge")
    args = parser.parse_args()
    
    setup_medical_knowledge_base()
