import asyncio

from sklearn.metrics import confusion_matrix
from inspect_ai.model import get_model
from inspect_ai.scorer import (
    mean, Scorer, Score, Target, scorer, accuracy, stderr, CORRECT, INCORRECT, metric, Metric, SampleScore
)
from inspect_ai.solver import TaskState

from rewards import NLI_LABELS
from prompt_templates import PROMPT_TEMPLATES, PROMPT_TEMPLATE_INFO
from utils import TASK_CRITERIA


async def judge_criterion(grader_model, state_mapping, task_criterion):
    prompt_template_args = {k: v.format(**state_mapping) for k, v in task_criterion["prompt_template_args"].items()}
    prompt_template_name = task_criterion["prompt_template_name"]
    score_prompt = PROMPT_TEMPLATES[prompt_template_name].render(**prompt_template_args)
    result = await grader_model.generate(score_prompt)
    match = PROMPT_TEMPLATE_INFO[prompt_template_name]["grade_pattern"].search(result.completion.lower())
    score = 0
    if match:
        model_answer = match.group("answer")
        if model_answer in task_criterion["target_answer_values"]:
            score = 1
    else:
        score = -1

    #
    return {"score": score, "completion": result.completion, "message": result.message, "grading_failure": score == -1}


@scorer(metrics={"*": [mean(), stderr()]})
def model_graded_by_tf(
        grading_criteria,
        conjunctive_criteria,
        model=None,

) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        state_mapping = {"p": state.metadata.get('premise', None),
                         "e": state.output.completion.strip(),
                         "h": state.metadata.get('hypothesis', None)}

        grader_model = model or get_model(role="grader")

        scores = await asyncio.gather(
            *[judge_criterion(grader_model, state_mapping, TASK_CRITERIA[criterion])
              for criterion in grading_criteria]
        )

        score_values = {k: v for k, v in zip(grading_criteria, scores)}

        # check for grading failures
        for k, v in score_values.items():
            if v["grading_failure"]:
                return Score.unscored(
                    answer=state.output.completion,
                    explanation=f"Grade not found in model output wrt {k}: {v['completion']}",
                    metadata={
                        "grader_model": grader_model.name,
                        "unscored_reason": f"grade_parsing_failure_wrt_{k}",
                        "grading": [TASK_CRITERIA[k]["prompt_template_name"],
                                    v["message"]]}
                )

        final_score_values = {
            k: v["score"] for k, v in score_values.items()
        }

        if conjunctive_criteria:
            for conjunctive_criterion in conjunctive_criteria:
                conjunctive_criterion_name = "__and__".join(conjunctive_criterion)
                scores_to_conjunct = [v["score"] for k, v in score_values.items() if k in conjunctive_criterion]
                if not scores_to_conjunct:
                    print(f"No scores found corresponding to: {conjunctive_criterion}")
                    continue
                final_score_values[conjunctive_criterion_name] = min(scores_to_conjunct)

        return Score(
            value=final_score_values,
            answer=state.output.completion,
            explanation="||\n||".join([v["message"].text for v in score_values.values()]),
            metadata={
                "grader_model": grader_model.name,
                "grading": ["||\n||".join(TASK_CRITERIA[k]["prompt_template_name"]
                                          for k in score_values.keys()),
                            "||\n||".join([v["message"].text
                                           for v in score_values.values()]),
                            ]}
        )

    return score


@metric
def confusion_matrix_metric() -> Metric:
    def calculate(scores: list[SampleScore]) -> dict[str, int]:
        y_true = [s.sample_metadata["target"] for s in scores]
        y_pred = [s.score.metadata["extracted_answer"] for s in scores]

        cm = confusion_matrix(y_true, y_pred, labels=NLI_LABELS)
        matrix_dict = {}
        for i, true_label in enumerate(NLI_LABELS):
            for j, pred_label in enumerate(NLI_LABELS):
                matrix_dict[f"True: {true_label} -> Pred: {pred_label}"] = int(cm[i][j])

        return matrix_dict

    return calculate


@scorer(metrics=[accuracy(), confusion_matrix_metric()])
def nli_label_matcher():
    async def score(state: TaskState, target: Target):
        completion = state.output.completion.strip()

        predicted = [False for i, _ in enumerate(NLI_LABELS)]

        for i, label in enumerate(NLI_LABELS):
            if label.lower() in completion.lower():
                predicted[i] = True

        predicted_label = "parsing_error" if sum(predicted) != 1 else NLI_LABELS[predicted.index(True)]
        return Score(value=CORRECT if state.metadata["target"] == predicted_label else INCORRECT,
                     answer=completion, metadata={"extracted_answer": predicted_label})

    return score


@scorer(metrics=[accuracy()])
def nli_label_matcher_control():
    # variant of nli_label_matcher without computing confusion matrix
    async def score(state: TaskState, target: Target):

        completion = state.output.completion.strip()

        predicted = [False for i, _ in enumerate(NLI_LABELS)]

        for i, label in enumerate(NLI_LABELS):
            if label.lower() in completion.lower():
                predicted[i] = True

        predicted_label = "parsing_error" if sum(predicted) != 1 else NLI_LABELS[predicted.index(True)]

        return Score(value=CORRECT if state.metadata["target"] == predicted_label else INCORRECT,
                     answer=completion, metadata={"extracted_answer": predicted_label})

    return score

