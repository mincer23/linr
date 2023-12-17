from llama_cpp import Llama

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

class web_llama:
    def __init__(self):

        self.template = """Question: {question}

        Answer: Let's work this out in a step by step way to be sure we have the right answer."""

        self.prompt = PromptTemplate(template=self.template, input_variables=["question"])

        # Callbacks support token-wise streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        self.n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
        self.n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

        # Make sure the model path is correct for your system!
        self.llm = LlamaCpp(
            model_path="/mnt/d/Projects/linr/llama-2-7b-chat.Q8_0.gguf",
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            callback_manager=self.callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
        )

    def process_text(self):
        llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
        question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
        llm_chain.run(question)