import torch
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# the configuration for quantization, or how to reduce weights in a way they
# fit on our gpu
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
#model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model_id = "lmlab/lmlab-mistral-1b-untrained"

model_4bit = AutoModelForCausalLM.from_pretrained( model_id, device_map="auto",quantization_config=quantization_config, )
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

# we create a huggingface llm pipeline
llm = HuggingFacePipeline(pipeline=pipeline)

#### Prompt
template = """<s>[INST] You are a helpful, interesting bartender with a background story.
You are in a metaverse where you can't actually sell anything, but you still perform engaging
conversations. Answer the questions maybe referencing
the context if deem it necessary for what is asked, although you do not have to reference
it always:
{context}
{question} [/INST] </s>
"""

question_p = """What would you like to sell today?"""
context_p = """ On August 10 said that its arm JSW Neo Energy has agreed to buy a portfolio of 1753 mega watt renewable energy generation capacity from Mytrah Energy India Pvt Ltd for Rs 10,530 crore.
Recently, the metaverse has been making strides towards better human/AI interaction and relations.
"""
prompt = PromptTemplate(template=template, input_variables=["question","context"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
response = llm_chain.run({"question":question_p,"context":context_p})
response