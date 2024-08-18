import argparse
import os
import pickle
import dotenv
import tiktoken
from typing import Dict
from hashlib import md5
from datetime import datetime, timedelta
import pandas as pd
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

dotenv.load_dotenv()


MODEL = "gpt-4o-mini"


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


def analyze_errors(df: pd.DataFrame) -> str:
    prompt_template = PromptTemplate.from_template("""
    You are an assistant that helps identify common causes of errors from logs.
    I will provide a DataFrame of my langchain agent calls that resulted in errors.
    Help me build out a report on the errors you see and any common causes of the errors.
    Here is the data:\n\n{df}
    """)

    llm = ChatOpenAI(model=MODEL)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(df=df.to_string())


def main(project_name: str):
    client = Client()
    df = get_project_errors(client, days=30, project_name=project_name)

    print(f"Total errors: {len(df)}")

    analysis = analyze_errors(df)
    print("\nError Analysis:")
    print(analysis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze errors for a LangSmith project")
    parser.add_argument("--project", help="Name of the LangSmith project to analyze")
    args = parser.parse_args()

    main(args.project)
