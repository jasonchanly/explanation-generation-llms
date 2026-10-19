from inspect_ai.solver import Solver, solver, generate

@solver
def prompt_template_jinja(template, **params) -> Solver:
    # modification of https://github.com/UKGovernmentBEIS/inspect_ai/blob/793a809a4cfefa5f1c37be5b7eada9a9df5f71ed/src/inspect_ai/solver/_prompt.py#L17
    # to use jinja2 templates rather than straightforward python strings
    async def solve(state, generate):
        kwargs = params | state.metadata
        rendered_prompt = template.render(**kwargs)
        state.user_prompt.text = rendered_prompt
        return state
    return solve

