import gradio as gr
import os
import boto3
from llama_index import GPTSimpleVectorIndex
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.agents import Tool
from langchain import OpenAI, LLMChain



s3 = boto3.resource('s3')
bucket_name = "notesinendocrinology"
bucket = s3.Bucket(bucket_name)
for obj in bucket.objects.filter(Prefix="comboindex.json"):
    combo_index_path = obj.key
    bucket.download_file(combo_index_path, "comboindex.json")

index = GPTSimpleVectorIndex.load_from_disk('comboindex.json')


def querying_db(query: str):
    response = index.query(query)
    return response


tools = [
    Tool(
        name="QueryingDB",
        func=querying_db,
        description="useful for when you need to answer questions from the database. The answer is given in bullet points.",
        return_direct=True
    )
]

prefix = "Give a detailed answer to the question"
suffix = """Give answer in bullet points

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad"]
)

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)


def get_answer(query_string):
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    result = agent_executor.run(query_string)

    return result


def qa_app(query):
    return get_answer(query)


inputs = gr.inputs.Textbox(label="Enter your question:")
output = gr.outputs.Textbox(label="Answer:")
iface = gr.Interface(fn=qa_app, inputs=inputs, outputs=output, title="Endo AI : Endocrine answering app by Dr. Om J Lakhani")

iface.launch()
