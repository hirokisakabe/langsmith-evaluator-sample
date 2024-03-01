import asyncio
import os
from operator import itemgetter

import langsmith
from dotenv import load_dotenv
from langchain.schema import output_parser
from langchain.smith import RunEvalConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult, run_evaluator
from langsmith.schemas import Example, Run

from lib import database


async def main():
    langsmith_client = langsmith.Client()

    dataset_name = os.getenv("DATASET_NAME")

    db = database.get_database()
    retriever = db.as_retriever(search_kwargs={"k": 4})
    template = """You are a helpful AI assistant. Answer the question in japanese based only on the following context:
{context}

Question: {input}
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = (
        {
            "context": itemgetter("input") | retriever,
            "input": itemgetter("input"),
        }
        | prompt
        | llm
        | output_parser.StrOutputParser()
    )

    @run_evaluator
    def is_empty(run: Run, example: Example | None = None):
        model_outputs = run.outputs["output"]
        score = not model_outputs.strip()
        return EvaluationResult(key="is_empty", score=score)

    eval_config = RunEvalConfig(
        evaluators=[
            # Correctness
            "qa",
            "context_qa",
            "cot_qa",
            # criteria
            RunEvalConfig.Criteria("conciseness"),
            RunEvalConfig.Criteria("relevance"),
            RunEvalConfig.LabeledCriteria(
                "correctness"
            ),  # correctnessだけはLabeledCriteriaしか使えない。他はどちらでも使える
            RunEvalConfig.Criteria("harmfulness"),
            RunEvalConfig.Criteria("maliciousness"),
            RunEvalConfig.Criteria("helpfulness"),
            RunEvalConfig.Criteria("controversiality"),
            RunEvalConfig.Criteria("misogyny"),
            RunEvalConfig.Criteria("criminality"),
            RunEvalConfig.Criteria("insensitivity"),
            RunEvalConfig.Criteria("depth"),
            RunEvalConfig.Criteria("creativity"),
            RunEvalConfig.Criteria("detail"),
            RunEvalConfig.Criteria(
                {"include_person_name": ("人名が含まれていますか？")}
            ),
            RunEvalConfig.LabeledCriteria(
                {
                    "refer_context": (
                        "referenceから得られる情報をもとに回答していますか？"
                    )
                }
            ),
            # distance
            "embedding_distance",
            "string_distance",
        ],
        # カスタム evaluator
        custom_evaluators=[is_empty],
        eval_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    )


    await langsmith_client.arun_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=chain,
        evaluation=eval_config,
        concurrency_level=5,
        verbose=True,
    )


if __name__ == "__main__":
    load_dotenv()

    asyncio.run(main())
