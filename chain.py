import logging

logger = logging.getLogger(__name__)


def llm_chain(llm, user_input, vector_retriever):
    try:
        context = vector_retriever.invoke(user_input)
        context_content = "\n\n".join(
            [f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}\n{doc.page_content}" for doc in context])
        logger.info(f'Found result : {context_content}')
        prompt = f"""Use only the context below to answer the question. If the answer is not in the context,
            say "I don't know".
        Context: {context_content}
        Question: {user_input}"""
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
