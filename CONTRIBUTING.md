# Contributing to FoodEval

Thanks for helping make FoodEval better. The benchmark only stays useful if the community runs new models on it, submits the results, flags bad data, and proposes tasks that cover gaps. This guide walks through all four.

If anything here is unclear, open an issue and we will fix the docs.

## The one rule that matters most

**Do not train on the benchmark data.** FoodEval is licensed CC-BY-NC-4.0 with an evaluation-only addendum. Using any of the files in `foodeval/data/` (or the copy on Hugging Face at [huggingface.co/datasets/latimal/foodeval](https://huggingface.co/datasets/latimal/foodeval)) as training data, fine-tuning data, distillation targets, or few-shot examples that get baked into a model is prohibited. The whole point of the benchmark is that no model has seen it during training. The moment someone trains on it, every score from that model becomes meaningless, and the leaderboard with it.

This applies to all of it: the retrieval corpora, the query relevance grades, the pair labels, and the cuisine labels. Evaluate on it, report numbers from it, cite it. Do not learn from it.

## Run the benchmark on a model

Install the package first (Python 3.10+). Pick the extra that matches what you want to evaluate.

```bash
# Local / Hugging Face sentence-transformers models
pip install "foodeval[local]"

# API models (OpenAI, AWS Bedrock)
pip install "foodeval[api]"

# Lexical baseline + REST API adapters (Cohere, Voyage, Gemini, Vertex AI), no torch, no SDKs
pip install foodeval
```

The leaderboard standardizes on d=384 (`--dim 384`) so models are compared at the same width. Run all 12 tasks for a leaderboard-eligible result. You can run a subset for quick checks with `--tasks`.

### Hugging Face and local models

Any sentence-transformers compatible model, by Hugging Face ID or local path. `--dim` truncates to the first N dimensions and re-normalizes (Matryoshka-style truncation).

```bash
# A model from the Hugging Face Hub
foodeval run --model BAAI/bge-m3 --dim 384 --output results/bge_m3_384.json

# A local fine-tuned checkpoint
foodeval run --model ./my-food-model --dim 384 --output results/my_food_model_384.json

# Just a couple of tasks while iterating
foodeval run --model intfloat/multilingual-e5-large --dim 384 --tasks food_search,indian_match
```

Dense leaderboard rows all run at d=384, whether or not the model was trained for truncation (see the README protocol note). A native-width run is welcome as an additional, non-ranked result file; note the dimension in your PR.

Note: the sentence-transformers adapter loads models with `trust_remote_code=True` (some architectures require it, e.g. GTE v1.5). Point it only at checkpoints you trust.

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
foodeval run --model openai:text-embedding-3-large --dim 384 --output results/openai_te3_large_384.json
```

The `--dim` value is passed through as the OpenAI `dimensions` parameter. For OpenAI-compatible endpoints (Azure, GitHub Models, a local server) set `OPENAI_BASE_URL` as well.

### Cohere

Uses the Cohere REST API directly. Pass the full model id with the version suffix.

```bash
export COHERE_API_KEY=...
foodeval run --model cohere:embed-v4.0 --dim 384 --output results/cohere_embed_v4_384.json
foodeval run --model cohere:embed-multilingual-v3.0 --dim 384 --output results/cohere_multilingual_v3_384.json
```

v3 models are truncated and re-normalized to `--dim`. v4 is requested at native width, then truncated and re-normalized client-side, the same final step the open-weight rows get.

### Voyage

Uses the Voyage embeddings REST API directly; the base install is enough. The default endpoint can be overridden with `VOYAGE_BASE_URL`.

```bash
export VOYAGE_API_KEY=...
foodeval run --model voyage:voyage-4-large --dim 384 --output results/voyage_4_large_384.json
```

Above 384, the narrowest native width Voyage serves is 512, so for `--dim 384` the adapter requests 512, then truncates and re-normalizes, mirroring the Titan treatment below. The adapter also paces requests for the free tier (3 requests per minute, about 21 seconds between calls); set `VOYAGE_MIN_INTERVAL=0` for paid keys.

### Gemini and Vertex AI

Both run on the base install. Vertex authenticates with Application Default Credentials (`gcloud auth application-default login`).

```bash
export GEMINI_API_KEY=...
foodeval run --model gemini:gemini-embedding-2 --dim 384 --output results/gemini_embedding_2_384.json

export GOOGLE_CLOUD_PROJECT=...
foodeval run --model vertex:gemini-embedding-001 --dim 384 --output results/vertex_gemini_001_384.json
```

### AWS Bedrock

Uses your standard AWS credential chain (env vars, `~/.aws/credentials`, or an instance role). Default region is `us-east-1`; override with `AWS_BEDROCK_REGION`.

```bash
foodeval run --model bedrock:cohere.embed-multilingual-v3 --dim 384 --output results/bedrock_cohere_v3_384.json
foodeval run --model bedrock:amazon.titan-embed-text-v2:0 --dim 384 --output results/bedrock_titan_v2_384.json
```

Titan only accepts dimensions in {256, 512, 1024}, so for `--dim 384` the adapter requests 512 natively, then truncates to 384 and re-normalizes. Cohere on Bedrock is truncated and re-normalized the same way as the direct API.

### Lexical baseline

A term-frequency baseline with no model download. Good sanity check and a reference point: on the current board it narrowly outscores two dense models, which is exactly the kind of result the benchmark exists to surface.

```bash
foodeval run --model lexical-tf
```

The `bm25` key is accepted as a legacy alias for the same adapter.

### Other useful commands

```bash
foodeval list                 # all 12 task names, types, and metrics
foodeval info food_search     # metadata for one task
foodeval leaderboard results/ # render a leaderboard from a directory of result JSONs
```

## Submit your results to the leaderboard

Leaderboard submissions come in as pull requests against `github.com/latimal/foodeval`. A complete submission is two things: the result JSON and a row in the leaderboard table.

1. **Run the full benchmark at d=384** (all 12 tasks). Partial runs are not eligible for the table.

   ```bash
   foodeval run --model <your-model> --dim 384 --output results/<your_model>_384.json
   ```

   Name the file after the model and dimension, lowercase with underscores, for example `results/voyage_4_large_384.json`.

2. **Add the result JSON** to the `results/` directory. The file is the raw output of the command above. Do not hand-edit the scores. The JSON carries `model_name`, `dimension`, `aggregate_score`, per-task `main_score` values, confidence intervals, and timings, which is what reviewers check against.

3. **Add your model's rows to the leaderboard tables** in `README.md`: the Overall table plus the Search, Matching, and Classification tables, each sorted by its final column descending, with the same four-decimal formatting as the existing rows. The numbers must match the JSON exactly. The easy way is to regenerate the tables with `foodeval leaderboard results/` and splice them in. If you would rather not touch the tables, say so in the PR and a maintainer will add the rows from your JSON. Maintainers sync the PyPI and Hugging Face copies of the tables at release time, so only `README.md` needs your edit.

4. **Put the exact command in the PR description**, so anyone can reproduce the run. Include:
   - the full `foodeval run ...` command you used
   - the model source (Hugging Face ID, API provider, or a link to the checkpoint)
   - the `foodeval --version` you ran
   - anything non-default: a custom adapter, a non-384 dimension, a special prompt or instruction prefix, the hardware if it matters

   A good PR description looks like this:

   ```
   Model: nomic-ai/nomic-embed-text-v1.5 (Hugging Face)
   Command: foodeval run --model nomic-ai/nomic-embed-text-v1.5 --dim 384 --output results/nomic_embed_v15_384.json
   foodeval version: 0.1.0
   Notes: default sentence-transformers adapter, Matryoshka truncation to 384.
   ```

5. **Confirm you did not train on the data** in the PR. One line is enough: "This model was not trained, fine-tuned, or distilled on FoodEval data." We take this on trust, and we will remove any result later shown to violate it. The package ships a checker so you can verify before you attest: `foodeval preflight --compare path/to/training_data/` hashes every benchmark surface and reports exact overlap.

A few things that get a submission bounced: scores in the table that do not match the JSON, a partial run presented as a full one, a result that cannot be reproduced from the command given, or a missing no-training statement.

### Submitting a model that needs a custom adapter

If your model is not reachable through the built-in adapters, implement the `EmbeddingAdapter` protocol (a `name` property, a `dimension` property, and an `encode(texts, batch_size, normalize) -> np.ndarray` method) and run it through `run_benchmark` in a short script. Include that script in your PR under `scripts/` so the run is reproducible. The README "Custom Adapters" section has a minimal template.

## Report a data issue

Found a mislabeled pair, a wrong relevance grade, a typo in the corpus, or a duplicate item? Please tell us. Clean data is the whole product.

Open an issue with the **Data issue** template (it applies the `data` label) and include:

- the task name (for example `cross_lingual_match`)
- the item `id` (every pair, query, and item has one, like `p001` or `q014`)
- what is wrong and what you think it should be
- a one-line reason, especially for label disputes where reasonable people might differ

We batch data fixes into a versioned release instead of patching files one at a time, so the benchmark stays a stable, citable target. Each task file carries a `version` field; a data fix bumps it and gets an entry in [CHANGELOG.md](CHANGELOG.md). If a fix is large enough to move scores, the changelog entry flags it so prior results can be read against the right data version.

For anything that looks like a systemic problem (a whole domain graded too generously, a script family handled inconsistently), open an issue and we will dig in rather than treat it as a single-item fix.

## Propose a new task

FoodEval grows by adding tasks that surface real failure modes generic benchmarks miss. If you have one in mind, open an issue with the **Task proposal** template before writing any code, so we can agree on scope first.

A strong proposal covers:

- **What capability it measures** and why current tasks do not already cover it. Gaps like "allergen cross-contamination reasoning" or "menu item to recipe retrieval" are worth filling.
- **The task type**: retrieval (NDCG@10), pair classification (Best F1), or classification (Macro F1). New metric types are possible but are a bigger lift, so flag that early.
- **Where the data comes from** and that it is clean enough to license under the benchmark's terms (CC-BY-NC-4.0 with the evaluation-only addendum). We cannot accept data scraped from sources we cannot redistribute.
- **A rough size**: how many queries, pairs, or items, and how many classes or domains.
- **A handful of example records** in the matching JSON schema (see the "Data Format" section of the README for the three schemas). Real examples make the difference between a proposal we can act on and one we cannot.

Once the design is agreed, the task slots in cleanly: a JSON file in `foodeval/data/`, one line in the task registry, and the existing retrieval / pair-classification / classification machinery handles the rest. We will help wire it up.

## Code contributions

Bug fixes, new adapters, and tooling improvements are welcome via PR. Clone the repo and install editable first: `pip install -e ".[all,dev]"`. Keep changes focused, run the test suite (`pytest`), and match the surrounding style (the project uses ruff with an 88-character line length). New adapters should follow the pattern of the existing ones: lazy-import the SDK, fail with a clear install message, support `--dim`, and cache embeddings through the shared cache helpers.

Thanks again for contributing. Every model run, data fix, and task idea makes the benchmark sharper.
