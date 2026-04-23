import asyncio
import json
import re
from agents.services.llm_service import ASI_MODEL, get_llm_client
from agents.services.tools import ALL_TOOL_HANDLERS


def _parse_text_tool_calls(text: str) -> list:
    """
    Parse <tool_call>...</tool_call> blocks that ASI1 sometimes emits as text
    instead of using the structured function-calling API.
    Returns list of {"name": str, "args": dict}.
    """
    calls = []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        block = match.group(1).strip()
        fn_match = re.match(r"^([a-zA-Z_]+)", block)
        if not fn_match:
            continue
        fn_name = fn_match.group(1)
        keys = re.findall(r"<arg_key>(.*?)</arg_key>", block)
        values = re.findall(r"<arg_value>(.*?)</arg_value>", block)
        calls.append({"name": fn_name, "args": dict(zip(keys, values))})
    return calls


def _strip_tool_call_markup(text: str) -> str:
    return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()


async def _execute_tool(tc) -> tuple:
    fn_name = tc.function.name
    try:
        fn_args = json.loads(tc.function.arguments)
    except json.JSONDecodeError:
        fn_args = {}
    handler = ALL_TOOL_HANDLERS.get(fn_name)
    result = await handler(**fn_args) if handler else f"Unknown tool: {fn_name}"
    return tc.id, result


async def _execute_text_tool_call(call: dict) -> str:
    handler = ALL_TOOL_HANDLERS.get(call["name"])
    if not handler:
        return f"Unknown tool: {call['name']}"
    return str(await handler(**call["args"]))


async def run_with_tools(query: str, system_prompt: str, tools: list) -> str:
    """
    Agentic tool-calling loop capped at 2 iterations.
    Handles both structured function-calling API and ASI1's text <tool_call> fallback.
    All tool calls in a single turn run in parallel via asyncio.gather.
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
        content = choice.message.content or ""

        # Structured function-calling API
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
                "content": _strip_tool_call_markup(content),
                "tool_calls": tool_calls_payload,
            })
            results = await asyncio.gather(*[_execute_tool(tc) for tc in choice.message.tool_calls])
            for tool_call_id, result in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(result),
                })

        # Text-based <tool_call> fallback (ASI1 quirk)
        elif _parse_text_tool_calls(content):
            text_calls = _parse_text_tool_calls(content)
            results = await asyncio.gather(*[_execute_text_tool_call(c) for c in text_calls])
            tool_results = "\n\n".join(
                f"[{c['name']} result]: {r}" for c, r in zip(text_calls, results)
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Tool results:\n{tool_results}\n\nNow provide your full analysis using these results."})

        else:
            return _strip_tool_call_markup(content)

    # Loop exhausted — force a text response without tools
    response = await client.chat.completions.create(model=ASI_MODEL, messages=messages)
    return _strip_tool_call_markup(response.choices[0].message.content or "")
