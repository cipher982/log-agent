import argparse
import os
import pickle
import re
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from hashlib import md5
from textwrap import dedent
from typing import Dict
from typing import List

import dotenv
import markdown2
import pandas as pd
import tiktoken
from atlassian import Confluence
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client
from tqdm import tqdm

dotenv.load_dotenv()


MODEL = "gpt-4o-mini"


def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))


def clean_whitespace(text: str) -> str:
    # Replace multiple newlines, tabs, or spaces with a maximum of two
    return re.sub(r"(\s)\1+", r"\1\1", text)


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
    df["input_tokens"] = df["input"].astype(str).apply(count_tokens)
    df["history_tokens"] = df["current_session_memory"].astype(str).apply(count_tokens)

    # Clean whitespace
    df["input"] = df["input"].astype(str).apply(clean_whitespace)
    df["current_session_memory"] = df["current_session_memory"].astype(str).apply(clean_whitespace)

    # Calculate token counts for query
    df["input_tokens"] = df["input"].astype(str).apply(count_tokens)
    df["history_tokens"] = df["current_session_memory"].astype(str).apply(count_tokens)

    drop_cols = ["current_session_memory"]
    df = df.drop(drop_cols, axis=1)

    return df


def analyze_single_error(row: pd.Series):
    prompt = PromptTemplate.from_template(
        dedent("""
        Analyze this single error log from a langchain agent call:

        Error: {error}
        Input: {input}
        Input Tokens: {input_tokens}
        History Tokens: {history_tokens}

        Provide a brief analysis of the error, including possible causes.

        some tips:
        - if the error is a chain error, try to find the root cause
        - some errors could be due to bad input, some due to bad instructions
        - sometimes it is token counts too high like people dumping a csv etc
        - when returning your analysis, be sure to include the input or a summary of it if too long,
          your analysis output is the only thing the reveiwer will see, so knowing about the input will help.    
    """)
    )

    llm = ChatOpenAI(model=MODEL)
    chain = prompt | llm

    inputs = row.to_dict()
    return chain.invoke(inputs)


def analyze_errors(df: pd.DataFrame) -> List[str]:
    analyses = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing errors"):
        analysis = analyze_single_error(row)
        analyses.append(analysis)
    return analyses


def summarize_analyses(analyses: List[str], project_name: str, start_date: str, end_date: str):
    prompt = PromptTemplate.from_template("""
    You are an assistant that helps summarize error analyses.
    I will provide a list of individual error analyses.
    Help me build out a comprehensive report summarizing the common causes of errors and any patterns you notice.

    Here are the individual analyses:

    {analyses}

    Please provide a summary of the main findings, grouping similar errors and highlighting any recurring issues.
                                          
    Some tips:
                                            
    - This is for technical maintainers of this AI agent code, so be helpful on the root causes that occur.
    - present a structured format for the summary, with a summary of the main findings, and a list of recurring issues.
                                          
    Format:
    Ensure that the formatting, including headers, bullet points, and lists,
    is consistent and adheres strictly to the markdown syntax provided below:

    ``` markdown
    # Langsmith Errors - {project_name} - {start_date} to {end_date}

    ## Summary of Main Findings
    Provide a high-level overview of the key issues identified during the analysis. 
    Group the findings by categories for clarity.

    ### 1. [Category Name]
    Provide a detailed explanation of the errors.

    Example Subheading: Explanation of the specific error or issue.
    Example Subheading: Explanation of another related error or issue.
    Include bullet points for listing examples, root causes, or contributing factors.
                                          
    ### 2. [Next Category Name]
    Continue with the next category in a similar format as above.

    ## Recurring Issues List

    Issue Summary: Brief description of the recurring issue.
    Issue Summary: Brief description of another recurring issue.
                                          
    ## Recommendations for Improvement

    Recommendation Summary: Actionable suggestion to address the identified issue.
    Recommendation Summary: Another actionable suggestion.
                                          
    ## Conclusion
    Summarize the importance of addressing the identified issues to improve overall system performance and reliability.
    ```

    Make sure to preserve this markdown structure, including all headers, subheaders, bullet points, 
    and other formatting elements, in every report.
    """)

    llm = ChatOpenAI(model=MODEL)
    chain = prompt | llm

    formatted_analyses = [f"Error {i+1}:\n{analysis.content}" for i, analysis in enumerate(analyses)]  # type: ignore
    inputs = {
        "analyses": "\n\n".join(formatted_analyses),
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

    base_title = f"Langsmith Errors ({MODEL})- {project_name} - {start_date} to {end_date}"
    title = base_title
    version = 1

    # Check if page exists and increment version number if necessary
    while confluence.page_exists(space_key, title):
        version += 1
        title = f"{base_title} v{version}"

    # Remove the triple backticks and "markdown" from the content
    cleaned_content = re.sub(r"```\s*markdown\s*\n", "", markdown_content)
    cleaned_content = re.sub(r"```\s*\n", "", cleaned_content)

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


def main(project_name: str, days: int = 30, debug: bool = False):
    start_date, end_date = get_date_range(days)

    client = Client()
    df = get_project_errors(client, days=30, project_name=project_name)
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

    individual_analyses = analyze_errors(df)

    summary = summarize_analyses(individual_analyses, project_name, start_date, end_date)
    summary_str = str(summary.content)
    print(f"\nError Analysis Summary ({start_date} to {end_date}):")

    # Save markdown content to a file
    markdown_file_path = f"./reports/{project_name}_error_analysis_{start_date}_{end_date}.md"
    with open(markdown_file_path, "w") as markdown_file:
        markdown_file.write(summary_str)
    print(f"Markdown file saved: {markdown_file_path}")

    create_confluence_page(summary_str, project_name, start_date, end_date)
    print("Confluence page created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze errors for a LangSmith project")
    parser.add_argument("--project", help="Name of the LangSmith project to analyze")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (sample first 2 rows)")
    args = parser.parse_args()

    main(args.project, debug=args.debug)
