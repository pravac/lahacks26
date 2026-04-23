import asyncio
import json
from agents.services.llm_service import ASI_MODEL, get_llm_client
from agents.services.tools import ALL_TOOL_HANDLERS


async def _execute_tool(tc) -> tuple:
    fn_name = tc.function.name
    try:
        fn_args = json.loads(tc.function.arguments)
    except json.JSONDecodeError:
        fn_args = {}
    handler = ALL_TOOL_HANDLERS.get(fn_name)
    result = await handler(**fn_args) if handler else f"Unknown tool: {fn_name}"
    return tc.id, result


async def run_with_tools(query: str, system_prompt: str, tools: list) -> str:
    """
    Agentic tool-calling loop capped at 2 iterations.
    All tool calls requested in a single LLM turn run in parallel via asyncio.gather.
    """
    client = get_llm_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    for _ in range(2):
        kwargs = {"model": ASI_MODEL, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_calls_payload = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": tool_calls_payload,
            })

            # Run all tool calls in parallel
            results = await asyncio.gather(*[_execute_tool(tc) for tc in choice.message.tool_calls])
            for tool_call_id, result in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(result),
                })
        else:
            return choice.message.content or ""

    # Loop exhausted — force a text response without tools so we always return something
    response = await client.chat.completions.create(model=ASI_MODEL, messages=messages)
    return response.choices[0].message.content or ""
