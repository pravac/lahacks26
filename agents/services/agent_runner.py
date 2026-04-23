import json
from agents.services.llm_service import ASI_MODEL, get_llm_client
from agents.services.tools import ALL_TOOL_HANDLERS


async def run_with_tools(query: str, system_prompt: str, tools: list) -> str:
    """
    Agentic tool-calling loop. The LLM decides which tools to call and when.
    Runs until the model produces a final text response (no more tool calls).
    Capped at 5 iterations to prevent runaway loops.
    """
    client = get_llm_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    for _ in range(5):
        kwargs = {"model": ASI_MODEL, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            # Append assistant message with tool call intent
            tool_calls_payload = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": tool_calls_payload,
            })

            # Execute each tool call and append results
            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                handler = ALL_TOOL_HANDLERS.get(fn_name)
                if handler:
                    result = await handler(**fn_args)
                else:
                    result = f"Unknown tool: {fn_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
        else:
            return choice.message.content or ""

    return choice.message.content or ""
