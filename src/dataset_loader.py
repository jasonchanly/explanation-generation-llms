import os
import json
from pathlib import Path
import pandas as pd
from datasets import Dataset
from inspect_ai.dataset import MemoryDataset, Sample
from prompt_templates import PROMPT_TEMPLATES

def get_datasets_base_dir():
    return Path("datasets") if "src" not in os.getcwd() else Path("../datasets")


def load_chaosnli(dset_split="mnli"):
    """

    :param dset_split:
    :return: df with cols ['id', 'label', 'label_weight', 'e_weight', 'n_weight', 'c_weight',
       'premise', 'hypothesis']
    """

    print(f"CURRENT WORKING DIRECTORY WHEN LOADING: {os.getcwd()}")
    if "mnli" in dset_split:
        fpath = get_datasets_base_dir() / "chaosnli/chaosNLI_mnli_m.jsonl"
    elif "snli" in dset_split:
        fpath = get_datasets_base_dir() / "chaosnli/chaosNLI_snli.jsonl"
    else:
        raise ValueError(f"Unexpected chaosnli subset: {dset_split}")

    data = []
    _label_mapping = {"e": "entailment", "n": "neutral", "c": "contradiction"}
    n_counter = 0
    with open(fpath) as f:
        for line in f:
            # Skip empty lines if they exist
            if line.strip():
                data_instance = json.loads(line)
                data.append({
                    "id": f"cn-{dset_split[0]}_{n_counter:06}_{data_instance['majority_label']}_{data_instance['uid']}",
                    "label": _label_mapping[data_instance["majority_label"]],
                    "label_weight": max(data_instance["label_dist"]),
                    "e_weight": data_instance["label_dist"][0], "n_weight": data_instance["label_dist"][1],
                    "c_weight": data_instance["label_dist"][2],
                    "premise": data_instance["example"]["premise"],
                    "hypothesis": data_instance["example"]["hypothesis"]})
                n_counter += 1

    return pd.DataFrame(data)


def load_chaosnli_mnli_for_judge_control(dset_split="mnli"):
    """

    :param dset_split:
    :return: df with cols ['id', 'label', 'label_weight', 'e_weight', 'n_weight', 'c_weight',
       'premise', 'hypothesis']
    """

    manual_df = pd.read_csv(get_datasets_base_dir()/"manually_written_adversarial_explanations.csv")
    manual_df["id"] = [x.replace("mnli_", "cn-m_") for x in manual_df["id"]]

    manual_entries = manual_df.to_dict("records")

    reshaped_entries = []

    for entry in manual_entries:
        reshaped_entries.append(
            {"id": f"{entry['id']}_effectiveness",
             "premise": f"{entry['premise']} {entry['explanation']}",
             "hypothesis": entry['hypothesis'],
             "label": "e"}
        )
        reshaped_entries.append(
            {"id": f"{entry['id']}_coherence",
             "premise": entry['premise'],
             "hypothesis": entry['explanation'],
             "label": "c"}
        )

    return pd.DataFrame(reshaped_entries)

def load_true_detective():
    fpath = "hf://buckets/jason-c/dataset-copies/true-detective/detective-puzzles.csv"

    source_df = pd.read_csv(fpath)
    source_df["id"] = [f"{i:03}" for i, _ in enumerate(source_df["case_name"])]
    return source_df


def load_custom_dataset(dataset_fpath, dset_format="reg_sft"):
    required_columns = {
        "reg_sft": ["id", "premise", "hypothesis", "completion"],
    }
    source_df = pd.read_csv(dataset_fpath)
    for required_column in required_columns[dset_format]:
        assert required_column in source_df.columns, f"{required_column} is not in data: {dataset_fpath}"

    return source_df


NAME_TO_FUNC_MAPPING = {
    "chaosnli": load_chaosnli,
    "true_detective": load_true_detective,
    "chaosnli_mnli_for_judge_control": load_chaosnli_mnli_for_judge_control,
    "custom_dataset": load_custom_dataset,
}


class GenericDataset:

    def __init__(self, dataset_name, dataset_filter_criteria=None, **dataset_kwargs):
        """

        :param dataset_name: ["abstentionbench", "chaosnli", "delta-causal", "inli", "probcopa", "anli"]
        :param dataset_kwargs: dset_split etc. to be passed to loader
        """
        self.dataset_name = dataset_name
        try:
            self.df = NAME_TO_FUNC_MAPPING[dataset_name](**dataset_kwargs)
        except KeyError:
            raise NotImplementedError(f"Dataset {dataset_name} not recognized.\n"
                                      f"Currently implemented: {list(NAME_TO_FUNC_MAPPING.keys())}")

        if dataset_filter_criteria:
            for k, v in dataset_filter_criteria.items():
                self.df = self.df[self.df[k] == v].copy()

    def to_inspectai_dataset(self, display_input_fields=("premise", "hypothesis",)):
        """
        For inpsect-ai use
        leave everything else as metadata
        :param display_input_fields:
        :param df:
        :param source_dset_format:
        :return:
        """
        def input_display_function(row_data):
            display_data = []
            for field in display_input_fields:
                try:
                    display_data.append(f"{field}: {row_data[field]}")
                except KeyError:
                    pass
            return "\n".join(display_data)

        if display_input_fields:
            return MemoryDataset([Sample(input=input_display_function(row), id=row["id"], metadata=row, target="") for row in self.df.to_dict("records")])
        else:
            return MemoryDataset([Sample(input="", id=row["id"], metadata=row, target="") for row in self.df.to_dict("records")])

    def to_inspectai_dataset_mcq(self):
        """

        :return:
        """
        if self.dataset_name == "true_detective":
            return MemoryDataset([Sample(input=row["mystery_text"], id=row["id"],
                                         choices=[ans[4:] for ans in row["answer_options"].split("; ")],
                                         target=row["answer"][1:2].upper(), metadata=row)
                                  for row in self.df.to_dict("records")])

        else:
            return NotImplementedError

    def to_hf_dataset(self, prompt_template_name,
                      prompt_template_kwargs=None,
                      completion_template_name=None,):
        # temporary workaround - to optimize if needed
        raw_dicts = [sample.model_dump()["metadata"] for sample in self.to_inspectai_dataset()]
        hf_dataset = Dataset.from_list(raw_dicts)

        formatting_fn_kwargs = {"prompt_template": PROMPT_TEMPLATES[prompt_template_name]}

        if prompt_template_kwargs:
            formatting_fn_kwargs.update(prompt_template_kwargs)

        if completion_template_name:
            formatting_fn_kwargs["completion_template"] = PROMPT_TEMPLATES[completion_template_name]

        formatted_dataset = hf_dataset.map(_format_instance, fn_kwargs=formatting_fn_kwargs)

        print(f"Example from formatted dataset: {formatted_dataset[0]}", flush=True)
        return formatted_dataset


def _format_instance(dataset_instance,
                     prompt_template,
                     completion_template=None,
                     **prompt_template_kwargs):
    instance_dict = {"prompt": [{"content": prompt_template.render(premise=dataset_instance["premise"],
                                                                   hypothesis=dataset_instance["hypothesis"],
                                                                   **prompt_template_kwargs).strip(), "role": "user"}]}
    if not completion_template:
        return instance_dict
    else:
        instance_dict["completion"] = [{"content": completion_template.render(completion=dataset_instance["completion"],
                                                                              **prompt_template_kwargs).strip(),
                                        "role": "assistant"}]
        return instance_dict


if __name__ == "__main__":
    pass