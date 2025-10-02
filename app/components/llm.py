# from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from app.config.config import HF_TOKEN,HUGGINGFACE_REPO_ID
from transformers import pipeline
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str=HUGGINGFACE_REPO_ID,hf_token:str=HF_TOKEN):
    try:
        logger.info("Loading LLM from HuggingFace")
        # Create HuggingFace pipeline with token for gated models
        generator = pipeline(
            task="text-generation",
            model=huggingface_repo_id,
            max_length=256,
            temperature=0.3,
            return_full_text=False,
            token=hf_token  # important for gated models
        )
        llm = ChatHuggingFace(
            pipeline=generator
        )

        logger.info("LLM loaded successfully...")

        return llm
    
    except Exception as e:
        error_message = CustomException("Failed to load a llm",e)
        logger.error(str(error_message))