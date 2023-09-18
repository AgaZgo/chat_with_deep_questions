from langchain.prompts import PromptTemplate


def build_prompt():
    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
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