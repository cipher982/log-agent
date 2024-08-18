import argparse
import os
import pickle
from datetime import datetime
from datetime import timedelta
from hashlib import md5
from typing import Dict

import dotenv
import pandas as pd
import tiktoken
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client

dotenv.load_dotenv()


MODEL = "gpt-4o-mini"


def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))


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
    # Calculate token counts for query
    df["input_tokens"] = df["input"].astype(str).apply(count_tokens)
    df["history_tokens"] = df["current_session_memory"].astype(str).apply(count_tokens)

    drop_cols = ["current_session_memory"]
    df = df.drop(drop_cols, axis=1)

    return df


def analyze_errors(df: pd.DataFrame) -> str:
    prompt = PromptTemplate.from_template("""
    You are an assistant that helps identify common causes of errors from logs.
    I will provide a DataFrame of my langchain agent calls that resulted in errors.
    Help me build out a report on the errors you see and any common causes of the errors.

    Some tips:
     - check history and input tokens, too many can reach context window ~128k
     - find common errors and group by error type
    
    Here is the data:\n\n{df}
    """)

    llm = ChatOpenAI(model=MODEL)
    chain = prompt | llm
    return chain.invoke({"df": df.to_string()})


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

    analysis = analyze_errors(df)
    print("\nError Analysis:")
    print(analysis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze errors for a LangSmith project")
    parser.add_argument("--project", help="Name of the LangSmith project to analyze")
    args = parser.parse_args()

    main(args.project)
