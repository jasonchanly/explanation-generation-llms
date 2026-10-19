import asyncio
import re

import numpy as np

from prompt_templates import PROMPT_TEMPLATES

NLI_LABELS_PREFIX = ("Ent", "Neu", "Con")
NLI_LABELS = ["entailment", "neutral", "contradiction"]

SERIAL_EXPL_REOBJ = re.compile(r"^\[?\d+\]?\. (?P<expl>.+)")


async def judge_reconciliation_by_nli(completion, premise, hypothesis, judge_client, judge_model_name,
                                      judge_generate_args):
    coherence_nli = PROMPT_TEMPLATES["judge/nli"].render(premise=premise, hypothesis=completion[-1]['content'])
    effectiveness_nli = PROMPT_TEMPLATES["judge/nli"].render(premise=f"{premise} {completion[-1]['content']}", hypothesis=hypothesis)

    prompts = [coherence_nli, effectiveness_nli]

    responses = await asyncio.gather(*[
        judge_client.chat.completions.create(
            model=judge_model_name, messages=[{"role": "user", "content": prompt}],
            **judge_generate_args,
        ) for prompt in prompts])

    coherence_prediction, effectiveness_prediction = responses[0].choices[0].message.content, responses[1].choices[0].message.content
    if "entailment" in coherence_prediction.lower():
        coherence_reward = 1
    elif "contradiction" in coherence_prediction.lower():
        coherence_reward = -1
    elif "neutral" in coherence_prediction.lower():
        coherence_reward = 0
    else:
        print("WARNING: JUDGE OUTPUT UNPARSED - coherence = 0")
        coherence_reward = 0

    if "entailment" in effectiveness_prediction.lower():
        effectiveness_reward = 1
    elif "contradiction" in effectiveness_prediction.lower():
        effectiveness_reward = -1
    elif "neutral" in effectiveness_prediction.lower():
        effectiveness_reward = 0
    else:
        print("WARNING: JUDGE OUTPUT UNPARSED - effectiveness = 0")
        effectiveness_reward = 0

    if min(coherence_reward, effectiveness_reward) == -1:
        return -1  # negative if any criterion hits a contradiction
    elif effectiveness_reward == 1:
        # only reward coherence if effective in the first place
        return np.mean([effectiveness_reward, coherence_reward])
    else:
        return 0


def make_reward_judge_reconciliation(judge_criteria, judge_client, judge_model_name):
    async def reward_judge_reconciliation(completions, premise, hypothesis, **kwargs):
        tasks = []
        for criterion_name, criterion_kwargs in judge_criteria.items():
            tasks.extend([
                REWARDS_JUDGE_REGISTER[criterion_name](_completions, _premise, _hypothesis, judge_client, judge_model_name,
                                                       **criterion_kwargs)
                for _completions, _premise, _hypothesis in zip(completions, premise, hypothesis)
            ])

        rewards = await asyncio.gather(*tasks)

        rewards_matrix = np.array(rewards).reshape(len(judge_criteria), -1).T

        # for now, taking the mean score over all criteria - we may devise more sophisticated aggregation later
        aggregated_rewards = rewards_matrix.mean(axis=1).tolist()
        return aggregated_rewards

    reward_judge_reconciliation.__name__ = "reward_judge_reconciliation"

    return reward_judge_reconciliation


REWARDS_REGISTER = {
    "judge_reconciliation": make_reward_judge_reconciliation,
}

REWARDS_JUDGE_REGISTER = {
    "judge_reconciliation_by_nli": judge_reconciliation_by_nli,
}
