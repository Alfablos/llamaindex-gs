"""
This script shows the workflow functionality from llama-index
Given 2 numbers (ints or floats) a model will perform the sum of the two,
while a second model will triple the result
"""

from llama_index.llms.deepseek import DeepSeek
from llama_index.core.agent.workflow import FunctionAgent, ToolCall
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step
)
from llama_index.utils.workflow import draw_all_possible_flows


llm = DeepSeek(model="deepseek-chat", temperature=0)

def add(x: float, y: float) -> float:
    """
    A function that adds two numbers.
    :param x: a float.
    :param y: another float.
    :return: a float, the result of adding.
    """
    return x + y


def triple(x: float) -> float:
    """
    A function that multiplies a number by 3.
    :param x: a float.
    :return: a float, the result of triplication.
    """
    return x * 3





class Added(Event):
    """
    Event containing the result of adding two numbers.

    Attributes:
        input (str): a string containing the 2 numbers that were added.
        output (str): the result of adding.
    """
    input: str
    output: str

class Multiplied(Event):
    """
    Event containing the result of multiplying two numbers.

    Attributes:
        input (str): a string representing a float.
        output (float): the result of multiplication.
    """
    input: str
    output: float


# Start => add() (send tool events if any) add_result => trilple() => triple_result => End
class MathWorkflow(Workflow):
    @step
    async def add(self, start_event: StartEvent, ctx: Context) -> Added:
        agent = FunctionAgent(
            llm=llm,
            tools=[add],
            system_prompt="You are an agent that can only perform sums between numbers. You're given a couple of numbers and you only output the result of their sum."
        )

        handler = agent.run(user_msg=start_event.numbers)

        async for event in handler.stream_events():
            # If a tool is called propagate the event to the main "event loop"
            if isinstance(event, ToolCall):
                ctx.write_event_to_stream(event)

        output = await handler

        return Added(input=start_event.numbers, output=str(output))

    @step
    async def triple(self, added: Added, ctx: Context) -> Multiplied:
        agent = FunctionAgent(
            llm=llm,
            tools=[triple],
            system_prompt="You are an agent that can only triple given input. You're given a number and you only output the result of its multiplication by three."
        )

        handler = agent.run(user_msg=added.output)

        async for event in handler.stream_events():
            # If a tool is called propagate the event to the main "event loop"
            if isinstance(event, ToolCall):
                ctx.write_event_to_stream(event)

        output = await handler

        return Multiplied(input=added.output, output=float(str(output)))

    @step
    async def finish(self, multiplied: Multiplied) -> StopEvent:
        return StopEvent(result=multiplied.output)




async def main():
    print("starting...")
    w = MathWorkflow(
        timeout=120,
        verbose=True
    )

    # draw_all_possible_flows(workflow=w, filename="possible_workflows.html")

    numbers = (3,5)

    handler = w.run(numbers=' '.join(
        map(lambda n: str(n), numbers)
    ))

    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            print(f'Tool called: {event}')
        if isinstance(event, StopEvent):
            result = event.result
            print(result)
            print('Type: ', type(result))





if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
