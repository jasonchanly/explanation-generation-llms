from inspect_ai import Task
from inspect_ai.solver import generate, multiple_choice
from inspect_ai.solver._multiple_choice import SINGLE_ANSWER_TEMPLATE as MCQ_SINGLE_ANSWER_TEMPLATE
from inspect_ai.scorer import choice
from inspect_ai import task

from dataset_loader import GenericDataset
from prompt_templates import PROMPT_TEMPLATES
import solvers
import scorers


@task
def generic_nli(dataset_name="chaosnli", prompt_template_name="judge/nli", dataset_args=None):
    dataset_args = {} if not dataset_args else dataset_args
    data = GenericDataset(dataset_name=dataset_name, **dataset_args)
    data.df["target"] = data.df["label"]
    return Task(
        dataset=data.to_inspectai_dataset(display_input_fields=("premise", "hypothesis",)),
        solver=[solvers.prompt_template_jinja(
            template=PROMPT_TEMPLATES[prompt_template_name]),
            generate(),
        ],
        scorer=scorers.nli_label_matcher(),
    )


@task
def generic_nli_control(dataset_name="chaosnli", prompt_template_name="judge/nli", dataset_args=None):
    dataset_args = {} if not dataset_args else dataset_args
    data = GenericDataset(dataset_name=dataset_name, **dataset_args)
    data.df["target"] = data.df["label"]
    return Task(
        dataset=data.to_inspectai_dataset(display_input_fields=("premise", "hypothesis",)),
        solver=[solvers.prompt_template_jinja(
            template=PROMPT_TEMPLATES[prompt_template_name]),
            generate(),
        ],
        scorer=scorers.nli_label_matcher_control(),
    )


@task
def generic_reconcile_contradictions(dataset_name="chaosnli",
                                     prompt_template_name="reconcile_contradiction/default-single",
                                     dataset_args=None) -> Task:
    dataset_args = {} if not dataset_args else dataset_args
    return Task(
        dataset=GenericDataset(dataset_name=dataset_name,
                               **dataset_args).to_inspectai_dataset(display_input_fields=("premise", "hypothesis",)),
        solver=[solvers.prompt_template_jinja(PROMPT_TEMPLATES[prompt_template_name]),
                generate(),
                ],
        scorer=scorers.model_graded_by_tf(grading_criteria=("reconciliatory_explanation_effectiveness",
                                                            "reconciliatory_explanation_coherence"),
                                          conjunctive_criteria=(("reconciliatory_explanation_effectiveness",
                                                                "reconciliatory_explanation_coherence"),
                                                                )),
    )

@task
def true_detective(dataset_args=None) -> Task:
    dataset_args = {} if not dataset_args else dataset_args
    return Task(
        dataset=GenericDataset(dataset_name="true_detective", **dataset_args).to_inspectai_dataset_mcq(),
        solver=multiple_choice(template=MCQ_SINGLE_ANSWER_TEMPLATE),
        scorer=choice(),
    )
