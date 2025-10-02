import os
from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH

logger=get_logger(__name__)

def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info("Loading existing vectorstore...")
            return FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True #Normally, FAISS may restrict loading certain saved indexes for security reasons.Setting this to True forces the library to load the index even if it could be “unsafe”, e.g., the file was saved with pickled Python objects.
            )
        else:
            logger.warning("No vector store found..")
    
    except Exception as e:
        error_message = CustomException("Failed to load vectorstore",e)
        logger.error(str(error_message))

#Creating new vectorstore function
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No chunks were found..")
        logger.info("Generating your new vectorestore")
        embedding_model=get_embedding_model()
        db=FAISS.from_documents(text_chunks,embedding_model)
        logger.info("Saving vector store")
        db.save_local(DB_FAISS_PATH)
        logger.info("Vectorstore saved successfully...")
        return db
    
    except Exception as e:
        error_message = CustomException("Failed to create new vector store",e)
        logger.error(str(error_message))

