from langchain.prompts import PromptTemplate


def build_prompt():
    template = """
    You are a helpful assistant with detailed knowledge about 'Deep Questions with Cal Newport' podcast. \
    Use the following transcripts of this podcast as a context (delimited by <ctx></ctx>) \
    and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    return prompt_template