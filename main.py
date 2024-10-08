import argparse
import asyncio
import os
import pickle
import re
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from hashlib import md5
from textwrap import dedent
from typing import Any
from typing import Dict
from typing import List

import dotenv
import markdown2
import pandas as pd
import tiktoken
from aiohttp import ClientSession
from atlassian import Confluence
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client
from tqdm import tqdm

dotenv.load_dotenv()


ERROR_MODEL = "gpt-4o-mini"
SUMMARY_MODEL = "gpt-4o"
MAX_TOKENS = 5000


def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))


def clean_whitespace(text: str) -> str:
    # Replace multiple newlines, tabs, or spaces with a maximum of two
    return re.sub(r"(\s)\1+", r"\1\1", text)


def truncate_long_text(text: str, max_tokens: int) -> str:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    keep_tokens = max_tokens // 2
    truncated_tokens = tokens[:keep_tokens] + tokens[-keep_tokens:]

    truncated_message = (
        f"{encoding.decode(truncated_tokens[:keep_tokens])}... "
        f"\n\n[Content truncated: {len(tokens) - max_tokens} tokens omitted]\n\n"
        f"...{encoding.decode(truncated_tokens[-keep_tokens:])}"
    )
    return truncated_message


def get_df_token_counts(df: pd.DataFrame) -> None:
    # Count tokens for each column
    token_counts: Dict[str, int] = {}
    for column in df.columns:
        token_counts[column] = df[column].astype(str).apply(count_tokens).sum()

    for column, count in token_counts.items():
        print(f"{column}: {count:,}")

    total_tokens = sum(token_counts.values())
    print(f"Total tokens in dataset: {total_tokens:,}\n")


def get_date_range(days: int) -> tuple:
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date


def format_metadata(metadata: Dict[str, Any]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in metadata.items() if v)


def get_project_errors(client: Client, days: int, project_name: str) -> pd.DataFrame:
    # Create a unique cache key based on the parameters
    cache_key = md5(f"{project_name}_{days}".encode()).hexdigest()
    cache_file = f"./exports/cache_{cache_key}.pkl"

    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Loading cached data for {project_name}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"Fetching data for {project_name}")
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    runs = list(
        client.list_runs(
            project_name=project_name,
            run_type="chain",
            start_time=start_time,
            error=True,
        )
    )

    df = pd.DataFrame(
        [
            {
                "name": run.name,
                "error": run.error,
                "latency": (run.end_time - run.start_time).total_seconds() if run.end_time else None,
                "prompt_tokens": run.prompt_tokens,
                "completion_tokens": run.completion_tokens,
                "total_tokens": run.total_tokens,
                "metadata": {
                    **run.extra.get("metadata", {}),  # type: ignore
                    **{f"{tag}": True for tag in run.tags},  # type: ignore
                },
                "start_time": run.start_time,
                **run.inputs,
                **(run.outputs or {}),
            }
            for run in runs
        ],
        index=[run.id for run in runs],
    )

    # Cache the dataframe
    with open(cache_file, "wb") as f:
        pickle.dump(df, f)

    return df


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Determine which column to use for input
    input_column = "input" if "input" in df.columns else "content"

    df["input_tokens"] = df[input_column].astype(str).apply(count_tokens)

    # Handle history if available
    if "current_session_memory" in df.columns:
        df["history"] = df["current_session_memory"].astype(str).apply(clean_whitespace)
        df["history_tokens"] = df["history"].astype(str).apply(count_tokens)
        df["history"] = df["history"].astype(str).apply(lambda x: truncate_long_text(x, max_tokens=500))
    else:
        df["history"] = ""
        df["history_tokens"] = 0

    # Clean whitespace and truncate input
    df[input_column] = df[input_column].astype(str).apply(clean_whitespace)
    df[input_column] = df[input_column].astype(str).apply(lambda x: truncate_long_text(x, max_tokens=MAX_TOKENS))

    # Rename 'content' to 'input' if necessary
    if input_column == "content":
        df = df.rename(columns={"content": "input"})

    # Drop unnecessary columns
    drop_cols = ["current_session_memory"] if "current_session_memory" in df.columns else []
    df = df.drop(drop_cols, axis=1)

    return df


async def analyze_single_error(row: pd.Series):
    prompt = PromptTemplate.from_template(
        dedent("""
        Analyze this single error log from a langchain agent call:

        Error: {error}
        Input: {input}
        Input Tokens: {input_tokens}
        {history_section}


        Provide a brief analysis of the error, including possible causes.

        some tips:
        - if the error is a chain error, try to find the root cause
        - some errors could be due to bad input, some due to bad instructions
        - sometimes it is token counts too high like people dumping a csv etc
        - when returning your analysis, be sure to include the input or a summary of it if too long,
          your analysis output is the only thing the reveiwer will see, so knowing about the input will help.    
    """)
    )

    llm = ChatOpenAI(model=ERROR_MODEL)
    chain = prompt | llm

    inputs = row.to_dict()
    if inputs["history"]:
        inputs["history_section"] = f"History: {inputs['history']}\nHistory Tokens: {inputs['history_tokens']}"
    else:
        inputs["history_section"] = "History: Not available"

    return await chain.ainvoke(inputs)


async def analyze_errors_async(df: pd.DataFrame) -> List[str]:
    async def process_row(row):
        return await analyze_single_error(row)

    analyses = []
    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests

    async with ClientSession() as _:
        tasks = []
        for _, row in df.iterrows():
            async with semaphore:
                task = asyncio.create_task(process_row(row))
                tasks.append(task)

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Analyzing errors"):
            analysis = await task
            analyses.append(analysis)

    return analyses


def summarize_analyses(df: pd.DataFrame, analyses: List[str], project_name: str, start_date: str, end_date: str):
    prompt = PromptTemplate.from_template("""
    You are an assistant that helps summarize error analyses.
    I will provide a list of individual error analyses.
    Help me build out a comprehensive report summarizing the common causes of errors and any patterns you notice.

    Here are the individual analyses:

    {analyses_with_metadata}

    Please provide a summary of the main findings, grouping similar errors and highlighting any recurring issues.

    Some tips:

    - This is for technical maintainers of this AI agent code, so be helpful on the root causes that occur.
    - Present a structured format for the summary, with a summary of the main findings, and a list of recurring issues.
    - Use the provided tags (like site_id and assistant_name) to give context instead of using generic error numbers.
    - Group errors by similar tags or characteristics when possible.
    - If certain tags (like specific site_ids or assistant names) appear frequently in errors, highlight in summary.

    Format:
    Ensure that the formatting, including headers, bullet points, and lists,
    is consistent and adheres strictly to the markdown syntax provided below:

    ``` markdown
    # Langsmith Errors - {project_name}
    #### Date Range: {start_date} to {end_date}

    ## Summary of Main Findings

    First you should provide a bulleted list with the following info:
    - Error Type
    - Count
    Then below use this format to group the main findings:

    ### 1. [Category Name]
    Provide a detailed explanation of the errors, using specific tags (assistant_name etc.) for context.

    Example Subheading: Explanation of the specific error or issue, mentioning relevant tags.
    Example Subheading: Explanation of another related error or issue, with tag context.
    Include bullet points for listing examples, root causes, or contributing factors.

    ### 2. [Next Category Name]
    Continue with the next category in a similar format as above.

    ## Recurring Issues List

    Issue Summary: Brief description of the recurring issue, mentioning relevant tags.
    Issue Summary: Brief description of another recurring issue, with tag context.

    ## Recommendations for Improvement

    Recommendation Summary: Actionable suggestion to address the identified issue.
    Recommendation Summary: Another actionable suggestion.

    ## Conclusion
    Summarize the importance of addressing the identified issues to improve overall system performance and reliability.
    ```

    Make sure to preserve this markdown structure, including all headers, subheaders, bullet points, 
    and other formatting elements, in every report.
    """)

    llm = ChatOpenAI(model=SUMMARY_MODEL, temperature=0.1)
    chain = prompt | llm

    formatted_analyses = []
    for i, analysis in enumerate(analyses):
        error_number = i + 1
        metadata_str = format_metadata(df.iloc[i]["metadata"])
        analysis_content = analysis.content  # type: ignore

        formatted_analysis = f"Error {error_number}:\n\nTags: {metadata_str}\n\nAnalysis: {analysis_content}"
        formatted_analyses.append(formatted_analysis)

    inputs = {
        "analyses_with_metadata": "\n\n".join(formatted_analyses),
        "project_name": project_name,
        "start_date": start_date,
        "end_date": end_date,
    }
    return chain.invoke(inputs)


def create_confluence_page(markdown_content: str, project_name: str, start_date: str, end_date: str):
    confluence = Confluence(
        url=os.getenv("CONFLUENCE_URL"),
        username=os.getenv("CONFLUENCE_USERNAME"),
        password=os.getenv("CONFLUENCE_API_TOKEN"),
    )

    space_key = os.getenv("CONFLUENCE_SPACE_KEY")
    parent_page_id = os.getenv("CONFLUENCE_PARENT_PAGE_ID")

    base_title = f"Errors: {project_name} - {start_date} to {end_date}"
    title = base_title
    version = 1

    # Check if page exists and increment version number if necessary
    while confluence.page_exists(space_key, title):
        version += 1
        title = f"{base_title} v{version}"

    # Remove the triple backticks and "markdown" from the content
    cleaned_content = re.sub(r"```\s*markdown\s*\n", "", markdown_content)
    cleaned_content = re.sub(r"```\s*\n", "", cleaned_content)
    cleaned_content = cleaned_content.rstrip("`")

    # Convert markdown to HTML
    html_content = markdown2.markdown(cleaned_content)

    confluence.create_page(
        space=space_key,
        title=title,
        body=html_content,
        parent_id=parent_page_id,
        representation="storage",
    )

    print(f"Confluence page created: {title}")


async def main(project_name: str, days: int, debug: bool = False):
    start_date, end_date = get_date_range(days)

    client = Client()
    df = get_project_errors(client, days=days, project_name=project_name)
    print(f"Total errors: {len(df)}\n")

    if debug:
        print("Debug mode: Sampling first 2 rows")
        df = df.head(2)

    print("=== Full token counts ===")
    get_df_token_counts(df)

    # Drop some columns
    cols_to_drop = ["latency", "prompt_tokens", "completion_tokens", "total_tokens"]
    df = df.drop(cols_to_drop, axis=1)
    print("=== Cleaned token counts ===")
    get_df_token_counts(df)

    df = process_dataframe(df)
    with open("processed.pkl", "wb") as f:
        pickle.dump(df, f)

    print("=== Processed token counts ===")
    get_df_token_counts(df)

    individual_analyses = await analyze_errors_async(df)

    summary = summarize_analyses(df, individual_analyses, project_name, start_date, end_date)
    summary_str = str(summary.content)
    print(f"\nError Analysis Summary ({start_date} to {end_date}):")

    # Save markdown content to a file
    markdown_file_path = f"./reports/{project_name}_error_analysis_{start_date}_{end_date}.md"
    with open(markdown_file_path, "w") as markdown_file:
        markdown_file.write(summary_str)
    print(f"Markdown file saved: {markdown_file_path}")

    create_confluence_page(summary_str, project_name, start_date, end_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze errors for a LangSmith project")
    parser.add_argument("--project", help="Name of the LangSmith project to analyze")
    parser.add_argument("--days", type=int, help="Number of days to analyze")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (sample first 2 rows)")
    args = parser.parse_args()

    asyncio.run(main(args.project, args.days, debug=args.debug))
