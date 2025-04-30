import time
from typing import Iterator, Optional
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGenerationChunk


class SlowGroq:
    def __init__(self, groq_model: ChatGroq, tokens_per_second: Optional[float] = None):
        """
        Wrap a ChatGroq instance to slow its streaming output.

        :param groq_model: Instance of ChatGroq to wrap.
        :param tokens_per_second: Desired output speed in tokens per second.
        """
        self.groq_model = groq_model
        self.tokens_per_second = tokens_per_second

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        token_interval = 1 / self.tokens_per_second if self.tokens_per_second else 0
        stream_iter = self.groq_model._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

        for chunk in stream_iter:
            yield chunk
            if token_interval:
                time.sleep(token_interval)

    def stream(
        self, messages: list[BaseMessage], **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        return self._stream(messages, **kwargs)


# Usage Example:
# groq_model = ChatGroq(model="llama-3.1-8b-instant")
# slow_groq = SlowGroq(groq_model, tokens_per_second=5)
#
# for chunk in slow_groq.stream(messages):
#     print(chunk.text, end="", flush=True)