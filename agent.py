import asyncio
import os
import sys
from typing import Any
import dotenv
from github import Github
from llama_index.core.agent.workflow import AgentOutput, FunctionAgent, AgentWorkflow, ToolCall, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

dotenv.load_dotenv()

# Command line arguments or environment variables
# args: agent.py $GITHUB_TOKEN $REPOSITORY $PR_NUMBER $OPENAI_API_KEY $OPENAI_BASE_URL
github_token = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else os.getenv("GITHUB_TOKEN")
repository = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else os.getenv("REPOSITORY")
pr_number_str = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else os.getenv("PR_NUMBER")
openai_api_key = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else os.getenv("OPENAI_API_KEY")
openai_base_url = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else os.getenv("OPENAI_BASE_URL")

if not repository:
    repository = "renbaum/recipes-api"

if "/" in repository:
    full_repo_name = repository
else:
    full_repo_name = f"renbaum/{repository}"

git = Github(github_token) if github_token else Github()
repo = git.get_repo(full_repo_name)


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Fetch details about the pull request given the PR number, including author, title, body, diff URL, state, head_sha, and commit SHAs."""
    pr = repo.get_pull(pr_number)
    commit_SHAs = []
    commits = pr.get_commits()
    for c in commits:
        commit_SHAs.append(c.sha)

    return {
        "author": pr.user.login if pr.user else "",
        "user": pr.user.login if pr.user else "",
        "title": pr.title if pr.title else "",
        "body": pr.body if (pr.body is not None and pr.body != "") else "No description provided.",
        "diff_url": pr.diff_url,
        "state": pr.state,
        "head_sha": pr.head.sha if hasattr(pr, "head") and pr.head else (commit_SHAs[-1] if commit_SHAs else ""),
        "commit_SHAs": commit_SHAs,
    }


def get_file_content(file_path: str) -> str:
    """Fetch the decoded contents of a file from the repository given the file path."""
    content = repo.get_contents(file_path)
    return content.decoded_content.decode("utf-8")


def get_pr_commit_details(commit_sha: str) -> list[dict[str, Any]]:
    """Retrieve details about a commit given the commit SHA, such as the files that changed, status, additions, deletions, changes, and patch."""
    commit = repo.get_commit(commit_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


async def add_context_to_state(ctx: Context, context: str) -> str:
    """Save gathered context in the current state."""
    try:
        if hasattr(ctx, "get"):
            current_state = await ctx.get("state")
        elif hasattr(ctx, "store"):
            current_state = await ctx.store.get("state", default={})
        else:
            current_state = {}
    except Exception:
        current_state = {}

    if current_state is None:
        current_state = {}

    current_state["gathered_contexts"] = context
    current_state["context"] = context

    if hasattr(ctx, "set"):
        await ctx.set("state", current_state)
    elif hasattr(ctx, "store"):
        await ctx.store.set("state", current_state)
    return "Context added to state."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Save draft comment in the current state."""
    try:
        if hasattr(ctx, "get"):
            current_state = await ctx.get("state")
        elif hasattr(ctx, "store"):
            current_state = await ctx.store.get("state", default={})
        else:
            current_state = {}
    except Exception:
        current_state = {}

    if current_state is None:
        current_state = {}

    current_state["draft_comment"] = draft_comment
    current_state["review_comment"] = draft_comment

    if hasattr(ctx, "set"):
        await ctx.set("state", current_state)
    elif hasattr(ctx, "store"):
        await ctx.store.set("state", current_state)
    return "Draft comment added to state."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Save final review in the current state."""
    try:
        if hasattr(ctx, "get"):
            current_state = await ctx.get("state")
        elif hasattr(ctx, "store"):
            current_state = await ctx.store.get("state", default={})
        else:
            current_state = {}
    except Exception:
        current_state = {}

    if current_state is None:
        current_state = {}

    current_state["final_review"] = final_review

    if hasattr(ctx, "set"):
        await ctx.set("state", current_state)
    elif hasattr(ctx, "store"):
        await ctx.store.set("state", current_state)
    return "Final review added to state."


def post_pr_review(pr_number: int, comment: str) -> str:
    """Post the final review comment to GitHub pull request given the PR number and comment."""
    pr = repo.get_pull(pr_number)
    pr.create_review(body=comment)
    return "Review posted successfully to GitHub."


pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
file_tool = FunctionTool.from_defaults(fn=get_file_content)
pr_commit_details_tool = FunctionTool.from_defaults(fn=get_pr_commit_details)
add_context_to_state_tool = FunctionTool.from_defaults(async_fn=add_context_to_state)
add_comment_to_state_tool = FunctionTool.from_defaults(async_fn=add_comment_to_state)
add_final_review_to_state_tool = FunctionTool.from_defaults(async_fn=add_final_review_to_state)
post_pr_review_tool = FunctionTool.from_defaults(fn=post_pr_review)

context_system_prompt = """You are the context gathering agent. When gathering context, you MUST gather: 
  - The details: author, title, body, diff_url, state, and head_sha; 
  - Changed files; 
  - Any requested for files; 
Once you gather the requested info, you MUST hand control back to the Commentor Agent.
"""

commentor_system_prompt = """You are the commentor agent that writes review comments for pull requests as a human reviewer would.
Follow these steps strictly:
1. When starting, you MUST always hand off to the ContextAgent first using the handoff tool to request the PR details, changed files, and repo files. Do not ask the user.
2. When the ContextAgent hands control back to you with the context:
   - Prepare a ~200-300 word review in markdown format detailing:
     - What is good about the PR?
     - Did the author follow ALL contribution rules? What is missing?
     - Are there tests for new functionality? If there are new models, are there migrations for them? (use the diff)
     - Are new endpoints documented? (use the diff)
     - Which lines could be improved upon? Quote these lines and offer suggestions.
     - Directly address the author (e.g. "Thanks for fixing this...").
   - Call the `add_comment_to_state` tool passing your drafted markdown review into `draft_comment`.
   - Call the `handoff` tool to hand off control to `ReviewAndPostingAgent`.
You MUST NEVER return a plain text response to the user. Always use `add_comment_to_state` and then `handoff` to `ReviewAndPostingAgent`.
"""

review_and_posting_system_prompt = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: 
   - Be a ~200-300 word review in markdown format. 
   - Specify what is good about the PR: 
   - Did the author follow ALL contribution rules? What is missing? 
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? 
   - Are there notes on whether new endpoints were documented? 
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? 
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. 
 When you are satisfied, you MUST call add_final_review_to_state to save the final review and call post_pr_review to post the review to GitHub.
"""

llm_kwargs = {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")}
if openai_api_key:
    llm_kwargs["api_key"] = openai_api_key
if openai_base_url:
    llm_kwargs["api_base"] = openai_base_url

llm = OpenAI(**llm_kwargs)

context_agent = FunctionAgent(
    name="ContextAgent",
    description="Gathers all the needed context from the repository for a pull request.",
    system_prompt=context_system_prompt,
    tools=[pr_details_tool, file_tool, pr_commit_details_tool, add_context_to_state_tool],
    llm=llm,
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent(
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment.",
    system_prompt=commentor_system_prompt,
    tools=[add_comment_to_state_tool],
    llm=llm,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

review_and_posting_agent = FunctionAgent(
    name="ReviewAndPostingAgent",
    description="Reviews the draft comment, requests rewrites if necessary, and posts the final review to GitHub.",
    system_prompt=review_and_posting_system_prompt,
    tools=[add_final_review_to_state_tool, post_pr_review_tool],
    llm=llm,
    can_handoff_to=["CommentorAgent"],
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review": "",
    },
)


async def main():
    pr_num = int(pr_number_str) if pr_number_str else 1
    query = f"Write a review for PR: {pr_num}"
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
