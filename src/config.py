import os
from dotenv import load_dotenv

# Load environment variables from the .env file located in the project root
load_dotenv()

# ==========================================
# 1. API Credentials
# ==========================================
# Retrieve API keys from environment variables for security
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") # <-- تمت إضافته هنا

# ==========================================
# 2. Model Selection Configuration
# ==========================================
# Choose the active provider: 'openai', 'google', 'anthropic', or 'deepseek'
ACTIVE_PROVIDER = os.getenv("ACTIVE_PROVIDER", "google").lower()

# Define the models capable of multimodal tasks (Vision + Text)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro") # <-- تمت إضافته هنا

# ==========================================
# 3. Directory Paths Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DOCS_DIR = os.path.join(BASE_DIR, "data", "raw-documents")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db_store")

# ==========================================
# 4. Processing Settings
# ==========================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Ensure the vector database directory exists before initialization
if not os.path.exists(VECTOR_DB_DIR):
    os.makedirs(VECTOR_DB_DIR)