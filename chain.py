import logging
from collections import defaultdict
logger = logging.getLogger(__name__)


def format_response(context, response):
    source_map = defaultdict(list)
    for doc in context:
        source = doc.metadata['source']
        page = doc.metadata['page']
        source_map[source].append(page)

    source_str = "\n\n".join([f"Source: {key}, Pages: {val}" for key,val in source_map.items()])
    if source_str:
        return f"{response.content} \n\n {source_str}"
    else:
        return f"{response.content}"


def llm_chain(llm, user_input, vector_retriever):
    try:
        logger.info(f"Embedding the user query: {user_input}")
        context = vector_retriever.invoke(user_input)
        context_content = "\n\n".join(
            [f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}\n{doc.page_content}" for doc in context])
        logger.info(f'Found context : {context_content}')
        prompt = f"""Use only the context below to answer the question. If the answer is not in the context,
say "I don't know".
Context: {context_content}
Question: {user_input}"""
        response = llm.invoke(prompt)
        logger.info(f"LLM Result: {response}")
        return format_response(context, response)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
