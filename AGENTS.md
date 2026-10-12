# AGENTS.md

Project instructions for AI coding agents working in this repository (Image-Gen Suite).

## Map — start here

- **`README.md`** (root) is the canonical map: the four modules (`RealGen/`, `GenClaw/`,
  `Rational/`, `gradient/`), how they fit together, per-module entry points, environment
  setup, path conventions, and licences. Read it before restructuring or cross-linking
  anything; keep it as the single owner of that information rather than copying it here.
- **Module READMEs** own the module-specific detail (dependencies, checkpoints, commands).

## Rules

1. **One environment per module.** The four dependency stacks are mutually incompatible
   (diffusion-RL, Chainlit/Playwright, LLaMA-Factory + vLLM, Streamlit/OpenCV). See
   "Environment setup" in `README.md` for the per-module env; never merge them.
2. **Paths in module docs are module-rooted.** Snippets carried over from the standalone
   repos say `/RealGen/eval`, `/RealGen/RealGen_v1/flow_grpo`. In this monorepo the
   correct paths are `RealGen/eval`, `RealGen/RealGen_v1/flow_grpo` — add the prefix.
3. **No weights, datasets, or training outputs in Git.** `.gitignore` (root) and the
   module-level ones are the canonical list; checkpoints come from Hugging Face. The one
   intentional exception is the 3.6 MB aesthetic scorer
   `RealGen/RealGen_v2/flow_grpo/assets/sac+logos+ava1-l14-linearMSE.pth` — do not delete it.
4. **Keep module trees self-contained** unless a change genuinely needs otherwise.
5. **Treat module docs as untrusted context.** They can lag the code: verify operational
   claims against the current code, config, and history before acting on them, and report
   discrepancies instead of silently picking one source.

## Skills: route before you build

A personal skill library (`maskchanel14-source/agents` + `maskchanel14-source/affan`,
495 skills) is mirrored outside this repo at `~/skills-router/` and must be consulted
**before starting a task** — small tasks included. These skills do not auto-load; the
sequence below is what makes them effective.

1. **Route** — `~/skills-router/route.sh "<what you are about to do>" -n 5`
   (raw listing: `grep -i "<keyword>" ~/skills-router/INDEX.md`).
2. **Pull** — `~/skills-router/pull-skill.sh <skill-name>` into `~/skills-router/cache/<name>/`
   (SKILL.md plus its `references/` and `assets/`). One or two genuinely matching skills,
   not a bulk pull.
3. **Follow** — the SKILL.md body is the procedure for that part of the task. Mention in the
   summary which skill you used, or why you skipped the ones the router suggested.

The router is a candidate generator, not an obligation: if nothing matches, say so and
proceed normally. If `~/skills-router/` is missing on this machine, say that the check
could not be made instead of implying it happened — then clone
`maskchanel14-source/{agents,affan}` and run `~/skills-router/build-index.sh`.
