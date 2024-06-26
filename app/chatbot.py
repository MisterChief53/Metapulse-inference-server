from typing import Any, Dict, Iterator, List, Optional
import re

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain.chains import LLMChain


class CustomChatModelAdvanced(BaseChatModel):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    model_name: str
    """The name of the model"""
    n: int
    """The number of characters from the last message of the prompt to be echoed."""
    llm: LLMChain
    """The LLMChain used to generate responses"""

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This receives a singular prompt and runs the model on that user message
        + the prompt to generate a response.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """

        # Replace this with actual logic to generate a response from a list
        # of messages.
        last_message = messages[-1]
        past_messages = ""
        ai_message = False
        for i in range(len(messages) - 1):
            if ai_message:
                past_messages += "You: "
                ai_message = False
            else:
                past_messages += "Human: "
                ai_message = True

            past_messages += messages[i].content[:] + '\n'
        print(past_messages)
        llm_response = self.llm.invoke({"question": last_message.content, "context": past_messages})

        # Find the index of the separator
        separator_index = llm_response["text"].find("</s>")

        text_after_separator = ""
        # If the separator is found, extract the text after it
        if separator_index != -1:
            text_after_separator = llm_response["text"][separator_index + len("</s>"):]
            # Truncate if the length is greater than 254 characters
            if len(text_after_separator) > 254:
                text_after_separator = text_after_separator[:254]
            # Remove newline characters using string replace
            text_after_separator = text_after_separator.replace("\n", "")
            text_after_separator = "AI: " + text_after_separator
            text_after_separator = re.sub(r'[^\x00-\x7F]+', '', text_after_separator)
            print(text_after_separator)

        message = AIMessage(
            content=text_after_separator,
            additional_kwargs={}, # we don't need another payload
            response_metadata={
                "time in seconds:": 3,
            },
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    # def _stream(
    #         self,
    #         messages: List[BaseMessage],
    #         stop: Optional[List[str]] = None,
    #         run_manager: Optional[CallbackManagerForLLMRun] = None,
    #         **kwargs: Any,
    # ) -> Iterator[ChatGenerationChunk]:
    #     """Stream the output of the model.
    #
    #     This method should be implemented if the model can generate output
    #     in a streaming fashion. If the model does not support streaming,
    #     do not implement it. In that case streaming requests will be automatically
    #     handled by the _generate method.
    #
    #     Args:
    #         messages: the prompt composed of a list of messages.
    #         stop: a list of strings on which the model should stop generating.
    #               If generation stops due to a stop token, the stop token itself
    #               SHOULD BE INCLUDED as part of the output. This is not enforced
    #               across models right now, but it's a good practice to follow since
    #               it makes it much easier to parse the output of the model
    #               downstream and understand why generation stopped.
    #         run_manager: A run manager with callbacks for the LLM.
    #     """
    #     last_message = messages[-1]
    #     tokens = last_message.content[: self.n]
    #
    #     for token in tokens:
    #         chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
    #
    #         if run_manager:
    #             # This is optional in newer versions of LangChain
    #             # The on_llm_new_token will be called automatically
    #             run_manager.on_llm_new_token(token, chunk=chunk)
    #
    #         yield chunk
    #
    #     # Let's add some other information (e.g., response metadata)
    #     chunk = ChatGenerationChunk(
    #         message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
    #     )
    #     if run_manager:
    #         # This is optional in newer versions of LangChain
    #         # The on_llm_new_token will be called automatically
    #         run_manager.on_llm_new_token(token, chunk=chunk)
    #     yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }
