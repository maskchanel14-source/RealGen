# Image-Gen Suite

A single monorepo collecting four complementary projects around **photorealistic
image generation, agentic image creation, reasoning-based reward models, and
AI-image forensics**.

Each project keeps its own source tree, dependencies, and original README; this
document is the map that ties them together.

> Migrated from four standalone repositories
> ([`GenClaw`](https://github.com/maskchanel14-source/GenClaw),
> [`RealGen`](https://github.com/maskchanel14-source/RealGen),
> [`Rational`](https://github.com/maskchanel14-source/Rational),
> [`gradient`](https://github.com/maskchanel14-source/gradient)) into plain
> sub-directories — no git submodules, no nested `.git` folders, one history.

---

## Table of contents

- [Modules at a glance](#modules-at-a-glance)
- [How the pieces fit together](#how-the-pieces-fit-together)
- [Repository layout](#repository-layout)
- [Module 1 — `RealGen/`](#module-1--realgen)
- [Module 2 — `GenClaw/`](#module-2--genclaw)
- [Module 3 — `Rational/`](#module-3--rational)
- [Module 4 — `gradient/`](#module-4--gradient)
- [Environment setup](#environment-setup)
- [Path conventions after the merge](#path-conventions-after-the-merge)
- [Large files, checkpoints, and .gitignore](#large-files-checkpoints-and-gitignore)
- [Licenses and attribution](#licenses-and-attribution)

---

## Modules at a glance

| Module | Folder | What it is | Core idea | Entry point |
|---|---|---|---|---|
| **RealGen** | `RealGen/` | Detector-guided RL post-training for text-to-image diffusion models (V1 + V2), plus the **RealBench** photorealism benchmark and evaluation scripts | Use an **AI-image detector as the reward** so the generator learns to escape detection → reduces artifacts, boosts realism | `RealGen/RealGen_v1/flow_grpo/…`, `RealGen/RealGen_v2/…`, `RealGen/eval/` |
| **GenClaw** | `GenClaw/` | **Code-driven agentic image generation**: an LLM agent that plans, sketches with code, then renders | *Think → sketch with code → render.* SVG/HTML/Python sketches make layout, counting, and text rendering explicit and debuggable | `chainlit run src/cc_genclaw/ui/chainlit_app.py -w` |
| **Rational** | `Rational/` | **RationalRewards** — a reasoning reward model + toolkit that scores visual generations through explicit multi-dimensional critique | Critique-before-score (PARROT) instead of one opaque scalar; optimizes both **train-time** (RL) and **test-time** (prompt refinement) | `Rational/*/run_*.sh` |
| **gradient** | `gradient/` | **Gradient-field / frequency-domain forensics** for telling real photos from AI-generated images, with Streamlit UIs | Natural-image statistics: gradient-field PCA (anisotropy, coherence), spectral slope, DCT/Benford, CFA, GLCM, ELA, noise kurtosis, EXIF, … fused by an ensemble scorer | `streamlit run gradient/app.py` |

---

## How the pieces fit together

The four projects cover the full loop of the "make synthetics look real" problem:

```
        ┌──────────────────────────────────────────────────────────────┐
        │                         GENERATION                           │
        │                                                              │
        │   GenClaw  ── agentic, code-first creation ──►  t2i / i2i    │
        │      (SVG/HTML drafts ─► render ─► VLM review ─► refine)     │
        └───────────────────────────┬──────────────────────────────────┘
                                    │ images
                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                    OPTIMIZATION / SCORING                    │
        │                                                              │
        │   RealGen  ── detector-guided GRPO reward for diffusion      │
        │      V1: static detectors (Forensic-Chat, OmniAID, LongCLIP) │
        │      V2: GAN-inspired dynamic discriminators (adversarial)   │
        │                                                              │
        │   Rational ── reasoning reward model (critique-before-score) │
        │      SFT ─► vLLM serving ─► RL training / prompt tuning      │
        └───────────────────────────┬──────────────────────────────────┘
                                    │ feedback signal
                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                       VERIFICATION                           │
        │                                                              │
        │   gradient ── gradient-field PCA + frequency forensics       │
        │   RealGen/benchmark ── RealBench (detector- & arena-scoring) │
        └──────────────────────────────────────────────────────────────┘
```

- **RealGen** and **Rational** are both *training-time* approaches: they supply
  reward signals (detector-based vs. reasoning-based) to RL post-training of
  diffusion models, and both ship test-time / evaluation tooling.
- **GenClaw** is the *generation-time* approach: no fine-tuning, instead the
  agent composes code drafts and image-model calls until the result is right.
- **gradient** is the *measurement* side: interpretable, non-neural forensics
  that can inspect any of the above outputs.

---

## Repository layout

```
.
├── README.md                 ← you are here (suite-level map)
├── .gitignore                ← suite-wide ignore rules
│
├── RealGen/                  ← detector-guided T2I RL + RealBench
│   ├── README.md
│   ├── RealGen_v1/           ← flow_grpo GRPO stack (FLUX / SD3.5) + reward-server
│   ├── RealGen_v2/           ← GAN-inspired online RL (Z-Image, paired real data)
│   ├── benchmark/            ← RealBench: real-img-benchmark, prompt.txt, eval_vs_real.py
│   ├── eval/                 ← GPT/Gemini evals + model-level reward evals
│   └── figures/              ← paper figures and comparisons
│
├── GenClaw/                  ← code-driven agentic image generation
│   ├── README.md
│   ├── src/cc_genclaw/       ← agent loop, tools, prompts, runtime, Chainlit UI
│   ├── public/, assets/, .chainlit/
│   ├── config.example.yaml, environment.yml, pyproject.toml
│
├── Rational/                 ← RationalRewards: reasoning reward model
│   ├── README.md
│   ├── rationalrewards_sft/          ← SFT on LLaMA-Factory
│   ├── reward_model_evaluation/      ← pairwise preference accuracy via vLLM
│   ├── diffusion_rl_training/        ← RL with RationalRewards feedback
│   ├── test_time_prompt_tuning/      ← Generate-Critique-Refine, no weight updates
│   └── assets/
│
└── gradient/                 ← gradient-field AI-image forensics
    ├── README.md, README_1.md
    ├── app.py, app_1.py, detect_ai.py, ai_detector_app.py
    ├── requirements.txt
    └── test_results/
```

---

## Module 1 — `RealGen/`

**Detector-guided RL for photorealistic text-to-image generation.** ECCV 2026
(Oral). Full details: [`RealGen/README.md`](RealGen/README.md).

### Architecture

| Component | Path | Purpose |
|---|---|---|
| **RealGen-V1** | `RealGen/RealGen_v1/flow_grpo/` | Flow-GRPO based GRPO post-training of diffusion models (FLUX.1-dev, SD3.5-large). Rewards come from *static* detectors. |
| **RealGen-V2** | `RealGen/RealGen_v2/` | GAN-inspired **online** RL: the generator trains against adversarially updatable discriminators over paired real-image + caption data. |
| **Reward servers** | `RealGen/RealGen_v1/flow_grpo/reward-server/` | Gunicorn-served reward models: **Forensic-Chat** (semantic detector), **OmniAID / OmniAID-DINO** (feature detector), **LongCLIP / CLIP** (alignment), plus HPSv2 / PickScore / Effort scorers. |
| **RealBench** | `RealGen/benchmark/` | Human-free photorealism benchmark: 1000+ real photographs + prompts, with **detector-scoring** and **arena-scoring** (`eval_vs_real.py`). |
| **Evaluation** | `RealGen/eval/` | Frontier-model judgments (`gpt-eval/`, Gemini) and per-reward-model evaluations (`model-eval/`: HPSv2/v3, LongCLIP, PickScore, Effort, Forensic-Chat, OmniAID). |

### Why V2 exists

V1 showed that a *static* detector reward is exploitable — the generator learns
to fool a frozen metric (reward hacking → eventual mode collapse). V2 replaces
text-only prompts with **paired data (real image + caption)** and a **hybrid
reward system with updatable discriminators**, so the objective keeps moving and
cannot be over-optimized.

### Quick start (V1 / V2)

```bash
# V1
cd RealGen/RealGen_v1/flow_grpo && conda create -n flow_grpo python=3.10.16 && pip install -e .
# reward server (separate env, conflicting deps by design)
cd RealGen/RealGen_v1/flow_grpo/reward-server && pip install -e .

# V2
cd RealGen/RealGen_v2/flow_grpo && pip install -e .
```

Weights/models to fetch separately: FLUX.1-dev, SD3.5-large, Z-Image,
[Realgen-detection-models](https://huggingface.co/lokiz666/Realgen-detection-models),
OmniAID, VisualQuality-R1, LongCLIP, and the RealGen-V2 real-image dataset.

---

## Module 2 — `GenClaw/`

**Code-driven agentic image generation** — an image agent that uses code as a
controllable visual canvas. Full details: [`GenClaw/README.md`](GenClaw/README.md).

### Architecture

A planner LLM runs in a tool-use loop. Before each task a `perception` sub-LLM
writes descriptive "painter notes" (intent + planning hints), then the planner
opens a `todo_write` plan and composes a tool pipeline:

| Layer | Path | Contents |
|---|---|---|
| Agent loop | `src/cc_genclaw/{agent,loop,llm,config,hooks,compaction,messages}.py` | planner loop, OpenAI-compatible client, context compaction |
| Perception | `src/cc_genclaw/perception/` | descriptive painter notes (not routing decisions) |
| Tools | `src/cc_genclaw/tools/` | `t2i` / `i2i`, `code_scene_draft` (SVG layout), `code_text_draft` (verbatim long text), `search` (Tavily), `reason`, `format_prompt`, `vlm_review`, `todo_write`, `tool_search` |
| Prompts | `src/cc_genclaw/prompts/` | `system.md`, perception prompt, per-tool cards |
| Runtime | `src/cc_genclaw/runtime/` | session dirs, artifact store, agent runner |
| UI | `src/cc_genclaw/ui/chainlit_app.py` | Chainlit web app |

Canonical chains (from the `todo_write` card):

- spatial / counting → `code_scene_draft → i2i`
- grounded entity → `search → format_prompt → i2i`
- scientific derivation → `reason → t2i`
- world-knowledge long text → `search → code_text_draft`

### Quick start

```bash
cd GenClaw
python -m venv .venv && source .venv/bin/activate && pip install -e .
playwright install chromium
cp config.example.yaml config.yaml     # single OpenAI-compatible gateway + model names
chainlit run src/cc_genclaw/ui/chainlit_app.py -w
```

Each run writes to `runs/YYYYMMDD-HHMMSS-XXXXXX/` — `trace.jsonl` plus every
intermediate artifact (SVG drafts, search references, final PNG).

---

## Module 3 — `Rational/`

**RationalRewards** — reasoning rewards that scale visual generation at both
training and test time. Full details: [`Rational/README.md`](Rational/README.md).

### Architecture

The reward model produces an explicit multi-dimensional critique *before*
emitting a score, using the **PARROT** recipe (Preference-Anchored
Rationalization): anchored rationale generation → consistency filtering →
distillation from preference-only data. This keeps instruction-following,
visual quality, composition, and plausibility as separate channels, which resists
reward hacking relative to scalar rewards.

| Sub-module | Path | Purpose |
|---|---|---|
| **SFT** | `Rational/rationalrewards_sft/` | Train the reward model on a Qwen3-VL-8B backbone (LLaMA-Factory stack); `run_sft.sh` |
| **RM evaluation** | `Rational/reward_model_evaluation/` | Pairwise preference accuracy against a vLLM-served checkpoint; `run_evaluation.sh` |
| **Diffusion RL** | `Rational/diffusion_rl_training/` | RL for T2I / image-editing generators with RationalRewards as reward (Edit-R1 + DiffusionNFT conventions); `run_rl_training.sh` |
| **Test-time tuning** | `Rational/test_time_prompt_tuning/` | Generate-Critique-Refine loop that improves outputs *without* parameter updates; `run_test_time_tuning.sh` |

### Quick start

```bash
cd Rational
bash rationalrewards_sft/run_sft.sh                                   # 1) SFT
vllm serve /path/to/checkpoint --port 6868                            # 2) serve
bash reward_model_evaluation/run_evaluation.sh                        # 3) evaluate
bash diffusion_rl_training/run_rl_training.sh                         # 4) RL
bash test_time_prompt_tuning/run_test_time_tuning.sh                  # 5) test-time
```

All scripts are environment-variable driven; replace `/path/to/...`
placeholders. Public checkpoints/datasets:
`TIGER-Lab/RationalRewards-8B-T2I`, `…-8B-Edit`, plus SFT/eval/RL datasets on
Hugging Face.

---

## Module 4 — `gradient/`

**Gradient-field PCA detector** — interpretable, non-neural forensics for
AI-generated images. Full details: [`gradient/README.md`](gradient/README.md)
and [`gradient/README_1.md`](gradient/README_1.md).

### Architecture

Classical natural-image statistics fused into an explainable verdict:

1. **Scharr/Sobel gradient field** on BT.709 luminance →
   `C = (1/N)·MᵀM` → eigendecomposition → **anisotropy** `ρ = λ₁/λ₂` and
   **coherence** `κ = ((λ₁−λ₂)/(λ₁+λ₂))²`. Real photos show coherent, high-ρ
   edges; diffusion outputs tend to be flatter / statistically unusual.
2. **Frequency domain**: power-spectrum slope and high-frequency ratio.
3. **Physics-inspired cues**: CFA (Bayer) correlations, chromatic aberration,
   noise-residual kurtosis, noise consistency across RGB channels.
4. **Structural cues**: DCT/Benford naturalness, GLCM texture, wavelet
   micro-texture kurtosis, edge-profile uniformity, JPEG ELA, upsampling-grid
   detection, LAB/HSV color anomalies, EXIF provenance flags.
5. **Ensemble decision engine**: weighted scoring with a human-readable report,
   plus a local SQLite **detection history** (`detections.db`).

| File | Role |
|---|---|
| `app.py` | Core gradient-field PCA analysis app |
| `app_1.py` | **SynthDetect Ultra** — full multi-feature forensic suite |
| `detect_ai.py` | Gradient-field analysis UI variant |
| `ai_detector_app.py` | Lightweight AI-image detection UI |
| `test_results/` | Example detections |

### Quick start

```bash
cd gradient
pip install -r requirements.txt
streamlit run app_1.py
```

---

## Environment setup

The four modules have **mutually incompatible dependency stacks** (diffusion RL
stacks, Chainlit/Playwright, LLaMA-Factory + vLLM, Streamlit/OpenCV). Use one
isolated environment per module — do not try to install everything together.

| Module | Suggested env | Notes |
|---|---|---|
| `RealGen/` | conda `flow_grpo`, Python 3.10.16 (separate `reward_server` env) | V2 needs a slightly different env than V1 |
| `GenClaw/` | `venv` + `pip install -e .` or conda `cc-genclaw` (`environment.yml`) | needs `playwright install chromium` |
| `Rational/` | separate envs for SFT / vLLM serving / RL | install upstreams first (LLaMA-Factory, vLLM, Edit-R1) |
| `gradient/` | any Python 3 with `gradient/requirements.txt` | CPU-only is fine |

## Path conventions after the merge

The modules were imported **unchanged**. Some module READMEs and shell scripts
use paths written as if the module were the repository root — for example
`cd /RealGen/RealGen_v1/flow_grpo` or `/RealGen/eval`. After the merge the
correct paths are prefixed with the suite folder:

| In module docs | In this monorepo |
|---|---|
| `/RealGen/RealGen_v1/flow_grpo` | `RealGen/RealGen_v1/flow_grpo` |
| `/RealGen/RealGen_v2` | `RealGen/RealGen_v2` |
| `/RealGen/eval` | `RealGen/eval` |

Relative asset links (e.g. `figures/…` inside `RealGen/README.md`, `../figures/…`
inside `RealGen/RealGen_v2/README.md`) keep working because the whole tree was
moved intact. Same for `Rational/assets/…`. If you copy a snippet, just add the
module folder prefix.

## Large files, checkpoints, and .gitignore

The suite-level [`.gitignore`](.gitignore) covers, for every module:

- Python bytecode and caches (`__pycache__/`, `*.py[cod]`), `*.egg-info/`,
  build artefacts;
- virtual environments (`.venv/`, `venv/`, `env/`, `.conda/`);
- test/lint caches (`.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.tox/`);
- **big model artefacts**: `*.pt`, `*.pth`, `*.bin`, `*.ckpt`, `*.safetensors`,
  `*.onnx`, `*.h5`, `*.npy`, `*.pkl`, `*.joblib` (the small 3.6 MB
  `sac+logos+ava1-l14-linearMSE.pth` aesthetic scorer shipped with RealGen-V2 is
  explicitly kept);
- training outputs (`/runs/`, `/outputs/`, `/logs/`, `/checkpoints/`,
  `/results/`, `wandb/`), local databases, datasets, and secrets
  (`.env`, `config.yaml`).

Also removed during the migration: a stray tracked `__pycache__/*.pyc`, a
`*.egg-info/` metadata folder inside `RealGen/RealGen_v1`, and the committed
`gradient/.DS_Store`.

**Weights are not in this repository.** Every checkpoint (FLUX/Z-Image/SD3.5,
detectors, reward models) must be downloaded from its Hugging Face page; the
per-module READMEs list them.

## Licenses and attribution

Each module is redistributed with its original files intact, including the
licence notices that shipped with it:

| Module | Licence files present | Origin |
|---|---|---|
| `RealGen/` | no repo-level `LICENSE`; third-party licences vendored with the code, e.g. `RealGen_v1/flow_grpo/reward-server/.../Long_CLIP/LICENSE` and `RealGen_v2/flow_grpo/Long_CLIP/LICENSE` (Apache-2.0) | `maskchanel14-source/RealGen` (built on [Flow-GRPO](https://github.com/yifan123/flow_grpo)) |
| `GenClaw/` | [`GenClaw/LICENSE`](GenClaw/LICENSE) — MIT | `maskchanel14-source/GenClaw` |
| `Rational/` | [`rationalrewards_sft/LICENSE`](Rational/rationalrewards_sft/LICENSE) and [`diffusion_rl_training/LICENSE`](Rational/diffusion_rl_training/LICENSE) — Apache-2.0 | `maskchanel14-source/Rational` (TIGER-AI-Lab RationalRewards; builds on LLaMA-Factory, vLLM, Edit-R1, DiffusionNFT, diffusers) |
| `gradient/` | no `LICENSE` file in the source repo | `maskchanel14-source/gradient` |

Because four projects with different licences now live in one repository, treat
each folder as its own licensing unit: the terms of a module apply to that
module's files, and upstream terms (model weights, datasets, base repositories)
still apply on top.

Citations for the underlying papers are in each module's README (RealGen:
arXiv 2512.00473; GenClaw: arXiv 2605.30248; RationalRewards: arXiv 2604.11626).
