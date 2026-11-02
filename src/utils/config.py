"""
Configuration management for the Medication Management AI system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Data paths
SYNTHETIC_PATIENTS_DIR = DATA_DIR / "synthetic_patients"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
MEDICAL_GUIDELINES_DIR = DATA_DIR / "medical_guidelines"
VOICE_SAMPLES_DIR = DATA_DIR / "voice_samples"

# Create data subdirectories
for dir_path in [SYNTHETIC_PATIENTS_DIR, VECTOR_DB_DIR, MEDICAL_GUIDELINES_DIR, VOICE_SAMPLES_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

# Model Configuration
# Priority: Groq > Grok > OpenAI
if GROQ_API_KEY:
    USE_GROQ = True
    LLM_MODEL = "llama-3.3-70b-versatile"  # Groq's fastest model
    LLM_BASE_URL = "https://api.groq.com/openai/v1"
    LLM_API_KEY = GROQ_API_KEY
elif GROK_API_KEY or XAI_API_KEY:
    USE_GROQ = False
    LLM_MODEL = "grok-beta"
    LLM_BASE_URL = "https://api.x.ai/v1"
    LLM_API_KEY = GROK_API_KEY or XAI_API_KEY
else:
    USE_GROQ = False
    LLM_MODEL = "gpt-4-turbo-preview"
    LLM_BASE_URL = None
    LLM_API_KEY = OPENAI_API_KEY

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VOICE_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# RAG Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.7

# Agent Configuration
MAX_ITERATIONS = 10
AGENT_TEMPERATURE = 0.7

# Voice Processing Configuration
SAMPLE_RATE = 16000
AUDIO_DURATION = 20  # seconds
STRESS_THRESHOLD = 0.6

# Simulation Configuration
DEFAULT_SIMULATION_DAYS = 14
DEFAULT_NUM_PATIENTS = 20
BASELINE_ADHERENCE_RATE = 0.45
TARGET_ADHERENCE_RATE = 0.78

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Debug mode
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
