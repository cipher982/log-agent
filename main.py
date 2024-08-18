import argparse
import os
import pickle
import re
from datetime import datetime
from datetime import timedelta
from hashlib import md5
from typing import Dict
from typing import List

import dotenv
import pandas as pd
import tiktoken
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
    start_time = datetime.utcnow() - timedelta(days=days)
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


def analyze_single_error(row: pd.Series) -> str:
    prompt = PromptTemplate.from_template("""
    Analyze this single error log from a langchain agent call:

    Error: {error}
    Input: {input}
    Input Tokens: {input_tokens}
    History Tokens: {history_tokens}

    Provide a brief analysis of the error, including possible causes.
    """)

    llm = ChatOpenAI(model=MODEL)
    chain = prompt | llm
    return chain.invoke(row.to_dict())


def analyze_errors(df: pd.DataFrame) -> List[str]:
    analyses = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing errors"):
        analysis = analyze_single_error(row)
        analyses.append(analysis)
    return analyses


def summarize_analyses(analyses: List[str]) -> str:
    prompt = PromptTemplate.from_template("""
    You are an assistant that helps summarize error analyses.
    I will provide a list of individual error analyses.
    Help me build out a comprehensive report summarizing the common causes of errors and any patterns you notice.

    Here are the individual analyses:

    {analyses}

    Please provide a summary of the main findings, grouping similar errors and highlighting any recurring issues.
    """)

    llm = ChatOpenAI(model=MODEL)
    chain = prompt | llm
    return chain.invoke({"analyses": "\n\n".join(analyses)})


def main(project_name: str):
    client = Client()
    df = get_project_errors(client, days=30, project_name=project_name)
    print(f"Total errors: {len(df)}\n")

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

    summary = summarize_analyses(individual_analyses)
    print("\nError Analysis Summary:")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze errors for a LangSmith project")
    parser.add_argument("--project", help="Name of the LangSmith project to analyze")
    args = parser.parse_args()

    main(args.project)
