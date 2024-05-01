import torch
from transformers import BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from chatbot import CustomChatModelAdvanced
from langchain_core.messages import AIMessage, HumanMessage

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate


# the configuration for quantization, or how to reduce weights in a way they
# fit on our gpu
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# model_id = "lmlab/lmlab-mistral-1b-untrained"
# model_id = "../models/Mistral-7B-Instruct-v0.1"
model_id = "/workspace/models/LocutusqueXFelladrin-TinyMistral248M-Instruct/"

print("getting model from its ID")
model_4bit = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                  quantization_config=quantization_config,
                                                  local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = pipeline(
    "text-generation",
    model=model_4bit,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=500,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

print("now creating pipeline")
# we create a huggingface llm pipeline
llm = HuggingFacePipeline(pipeline=pipeline)

#### Prompt
template = """<s>[INST] You are a helpful, interesting bartender with a background story.
You are in a metaverse where you can't actually sell anything, but you still perform engaging
conversations. Answer the questions maybe referencing
the context if deem it necessary for what is asked, although you do not have to reference
it always. Now, I will show you your message history so far (the first message was made by a human), Just generate single brief reply Do not add "You".
{context}
The human's next message:
{question} [/INST] </s>
"""

question_p = """What would you like to sell today?"""
context_p = """ On August 10 said that its arm JSW Neo Energy has agreed to buy a portfolio of 1753 mega watt renewable energy generation capacity from Mytrah Energy India Pvt Ltd for Rs 10,530 crore.
Recently, the metaverse has been making strides towards better human/AI interaction and relations.
"""
prompt = PromptTemplate(template=template, input_variables=["question", "context"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

advanced_model = CustomChatModelAdvanced(n=3, model_name="my_custom_model", llm=llm_chain)

response = advanced_model.invoke(
    [
        HumanMessage(content="hello!"),
        AIMessage(content="Hi there human!"),
        HumanMessage(content="How was your day?"),
    ]
)

response = response.content

# Find the index of the separator
separator_index = response.find("</s>")

# If the separator is found, extract the text after it
if separator_index != -1:
  text_after_separator = response[separator_index + len("</s>"):]
  print(text_after_separator)
else:
  print("Separator not found")



app = FastAPI(
    title="Inference Server",
    version="1.0",
    description="An api to interact with our LLM"
)

"""
This is a simple FastAPI server that serves the LLM model. It has a single endpoint
"""
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

prompt = ChatPromptTemplate.from_template("User's message: {message}")
# Edit this to add the chain you want to add
add_routes(
    app,
    prompt | advanced_model,
    path="/chat"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
