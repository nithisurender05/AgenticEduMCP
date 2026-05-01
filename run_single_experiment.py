import os
import json
import asyncio
import time
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
import random

# Ensure API key is present
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set your GEMINI_API_KEY environment variable.")

# Initialize modern Gemini client
client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.1-flash-lite-preview"

# Helper function to handle rate limits
async def safe_generate(**kwargs):
    last_error = None

    for i in range(6):
        try:
            return await asyncio.to_thread(
                client.models.generate_content,
                **kwargs
            )

        except (ServerError, ClientError) as e:
            last_error = e

            # exponential backoff + jitter
            wait = min(60, (2 ** i) + random.random())
            print(f"[retry {i}] waiting {wait:.1f}s due to error: {e}")
            await asyncio.sleep(wait)

    raise RuntimeError(f"Gemini failed after retries: {last_error}")

# Running the experiment
async def run_experiment(student):
    server_params = StdioServerParameters(
        command="python3",
        args=["canvas_mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\n" + "="*50)
            print("Starting experiment ...")
            print("="*50)

            # Baseline prompt with no context or tools
            basic_prompt = (
                f"I'm {student['name']} (id {student['id']}). "
                f"Help me study for my MATH-022 (Calculus II) final. "
                f"I don't know what to focus on."
            )

            grades_text = "\n".join([
                f"- {g['assignment']}: {g['score']}"
                for g in student["courses"][0]["grades"]
            ])

            # Student prompt with context (but no tools)
            contextual_prompt = f"""
            I'm {student['name']} (id {student['id']}). 
            My current scores are:
            {grades_text}

            Help me study for my MATH-022 final.
            """

            # CONDITION 1: BASELINE LLM (NO MCP TOOLS)
            print("\n[CONDITION 1] BASELINE: Standard LLM (No MCP Tools)")
            print(f"User: {basic_prompt}")

            baseline_response = await safe_generate(
                model=MODEL_NAME,
                contents=basic_prompt
            )

            print(f"\nBaseline LLM Response:\n{baseline_response.text}")
            print("-" * 50)

            # CONDITION 2: CONTEXTUAL LLM (Standard LLM with Context)
            print("\n[CONDITION 2] CONTEXTUAL: Standard LLM (With Context)")
            print(f"User: {contextual_prompt}")
            contextual_response = await safe_generate(
                model=MODEL_NAME,
                contents=contextual_prompt
            )
            print(f"\nContextual LLM Response:\n{contextual_response.text}")

            # CONDITION 3: AGENTIC LLM (With Mock Canvas MCP)
            print("\n[CONDITION 3] AGENTIC: Agentic LLM (With Canvas MCP)")

            # -------- Fetch MCP tools --------
            mcp_tools_response = await session.list_tools()

            def clean_schema(schema_dict):
                if isinstance(schema_dict, dict):
                    schema_dict = dict(schema_dict)
                    schema_dict.pop("title", None)
                    schema_dict.pop("default", None)

                    if "type" in schema_dict and isinstance(schema_dict["type"], str):
                        schema_dict["type"] = schema_dict["type"].upper()

                    for k, v in list(schema_dict.items()):
                        schema_dict[k] = clean_schema(v)

                elif isinstance(schema_dict, list):
                    return [clean_schema(i) for i in schema_dict]

                return schema_dict

            gemini_tools = []
            for tool in mcp_tools_response.tools:
                clean_params = clean_schema(json.loads(json.dumps(tool.inputSchema)))
                gemini_tools.append({
                    "functionDeclarations": [{
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": clean_params
                    }]
                })

            print(f"--> Success: Agent discovered {len(gemini_tools)} Canvas tools.")

            # -------- Agent loop --------
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=basic_prompt)]
                )
            ]
            steps = 0
            max_steps = 5
            while True:
                steps += 1
                if steps >= max_steps:
                    print("Reached max steps. Ending experiment.")
                    break
                response = await safe_generate(
                    model=MODEL_NAME,
                    contents=contents,
                    config={"tools": gemini_tools}
                )

                candidate = response.candidates[0]
                parts = candidate.content.parts

                function_call = None
                for p in parts:
                    if hasattr(p, "function_call") and p.function_call:
                        function_call = p.function_call
                        break

                #  If no tool call → we're done
                if not function_call:
                    print(f"\nFinal Grounded AI Response:\n{response.text}")
                    print("="*50)
                    break

                tool_name = function_call.name
                tool_args = dict(function_call.args)

                print(f"--> ⚙️ Gemini autonomously decided to call: {tool_name}")
                print(f"-->    Arguments: {tool_args}")

                # -------- Execute MCP tool --------
                mcp_result = await session.call_tool(tool_name, tool_args)

                result_text = (
                    mcp_result.content[0].text
                    if mcp_result.content else "No data returned."
                )

                print(f"-->    Server returned data length: {len(result_text)} chars")

                # -------- Append model call + tool result --------
                contents.append(candidate.content)

                contents.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=tool_name,
                                    response={"result": result_text}
                                )
                            )
                        ]
                    )
                )

            await asyncio.sleep(0.5)
            
            # CONDITION 4: FULL CONTEXT LLM
            print("\n[CONDITION 4] FULL CONTEXT LLM ")
            full_context_prompt = f"""
                I'm {student['name']} (id {student['id']}).
                Here is my COMPLETE student profile:
                Grades: {grades_text}
                Feedback: {student['courses'][0]['feedback']}
                Syllabus: {student['courses'][0]['syllabus']}
                Help me study for my MATH-022 final.
            """

            full_context_response = await safe_generate(
                model=MODEL_NAME,
                contents=full_context_prompt
                
            )

            
    return {
        "student_id": student["id"],
        "baseline": baseline_response.text,
        "contextual": contextual_response.text,
        "full_contextual": full_context_response.text,
        "agentic": response.text  # final grounded response
    }

def load_students(path="mock_canvas_data.json"):
    with open(path, "r") as f:
        data = json.load(f)

    # extract correct field
    return data["student_profiles"]


def get_student_by_id(students, student_id):
    for s in students:
        if str(s["id"]) == str(student_id):
            return s
    raise ValueError(f"Student {student_id} not found")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    students = load_students()

    # choose which student to run
    TARGET_STUDENT_ID = "404"   

    student = get_student_by_id(students, TARGET_STUDENT_ID)

    result = asyncio.run(run_experiment(student))

    # print output
    print("\n\n========== FINAL OUTPUT ==========\n")
    print(json.dumps(result, indent=2))

    #  save to file
    with open(f"single_run_{TARGET_STUDENT_ID}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to single_run_{TARGET_STUDENT_ID}.json")