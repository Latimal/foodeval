# FoodEval

[![PyPI](https://img.shields.io/pypi/v/foodeval)](https://pypi.org/project/foodeval/) [![Python](https://img.shields.io/pypi/pyversions/foodeval)](https://pypi.org/project/foodeval/) [![License](https://img.shields.io/badge/license-CC--BY--NC--4.0_eval--only-blue)](https://github.com/latimal/foodeval/blob/main/LICENSE)

**The food-domain benchmark for text embedding models.**

Mainstream embedding benchmarks (MTEB, BEIR) contain no menu- or dish-level evaluations. The closest thing, NFCorpus, is medical-literature retrieval seeded from nutrition topics. Models that do well on those benchmarks stumble on food text. On our leaderboard, OpenAI's text-embedding-3-large, requested at d=384 through its native dimensions parameter, returns "egg white omelette" as its top hit for "egg free options" and "lassi" for "lactose free". Both queries score an NDCG@10 of 0.0, verifiable in its shipped result file. No model reaches 0.24 NDCG@10 on diet and allergen search, ours included.

On cross-lingual matching, two popular English-focused open-weight models (Nomic embed-text v1.5, Alibaba GTE-large-en v1.5) sit exactly at the all-positive floor, the F1 for calling every pair a match. That part is expected. The sharper result: the three dedicated multilingual models on the board (BAAI/bge-m3, multilingual-e5-large, Cohere multilingual-v3) clear the term-frequency baseline on that task by under 2.5 points of F1, while Voyage 4 Large, the strongest general-purpose model there, clears it by more than 11. bge-m3 prescribes no prefixes, so its margin is clean; the e5 row runs without its card's prefixes (see the protocol note below). FoodEval measures what generic benchmarks miss.

12 tasks. 5,868 evaluation examples. 26 menu classes. Graded relevance judgments over production menu data.

Published by [Latimal](https://latimal.com).

## Tasks

| Task | Type | Metric | Examples | What it measures |
|------|------|--------|----------|-----------------|
| Food Search | Retrieval | NDCG@10 | 178 queries, 974 docs | Ranked retrieval across Indian, global, and beverage menus |
| Concept Search | Retrieval | NDCG@10 | 44 queries, 301 docs | Abstract food concept queries ("warm comfort food", "barbecue dish") |
| Diet/Allergen Search | Retrieval | NDCG@10 | 35 queries, 245 docs | Diet- and allergen-specific queries ("celiac friendly", "shellfish allergy safe") |
| Noisy Search | Retrieval | NDCG@10 | 46 queries, 304 docs | Misspelled and abbreviated queries ("bibimbab", "bbq burgeer") |
| Indian Match | Pair classification | Best F1 | 498 pairs | Same-dish detection for Indian cuisine menu items |
| Global Match | Pair classification | Best F1 | 318 pairs | Same-dish detection across global (non-Indian) cuisines |
| Beverage Match | Pair classification | Best F1 | 295 pairs | Same-drink detection across teas, coffees, juices, smoothies, and cocktails |
| Bakery Match | Pair classification | Best F1 | 336 pairs | Same-item detection for bakery and dessert items |
| Portion Size | Pair classification | Best F1 | 228 pairs | Distinguishing portion/size variants from genuinely different items |
| Noisy Menu Match | Pair classification | Best F1 | 323 pairs | Same-dish detection with markup noise, casing, and symbol artifacts |
| Cross-Lingual Match | Pair classification | Best F1 | 514 pairs | Cross-script, romanized, and bilingual dish matching |
| Cuisine Classification | Classification | Macro F1 | 3,053 items, 26 classes | Linear probe cuisine classification from frozen embeddings |

## Quick Start

```bash
pip install "foodeval[local]"
```

Requires Python 3.10+. The `[local]` extra pulls in sentence-transformers (plus torch and einops). Bare `pip install foodeval` is enough for the lexical baseline and the REST-based API adapters (Cohere, Voyage, Gemini, Vertex AI); the OpenAI and Bedrock adapters need `pip install "foodeval[api]"`.

Evaluate any sentence-transformers model in a few lines:

```python
from foodeval.evaluate import run_benchmark
from foodeval.adapters.sentence_transformer import SentenceTransformerAdapter

adapter = SentenceTransformerAdapter("sentence-transformers/all-MiniLM-L6-v2", truncate_dim=384)
result = run_benchmark(adapter)
print(result.to_markdown())
```

Or from the command line:

```bash
foodeval run --model sentence-transformers/all-MiniLM-L6-v2 --dim 384
```

The MiniLM model is a 90 MB download, so this gives a first result table in minutes; swap in any leaderboard model, e.g. `BAAI/bge-m3`, when you want the real comparison.

## Leaderboard

Nine dense models evaluated at d=384, plus a lexical baseline at its native term-frequency dimension. FoodEval Score = unweighted mean across all 12 tasks. The built-in adapters reach more providers than the launch board covers; rows for Gemini, Vertex AI, and Bedrock models come in as community submissions (see Contributing Results). Browse the board interactively at the [FoodEval leaderboard Space](https://huggingface.co/spaces/latimal/foodeval-leaderboard); merged submissions appear there automatically.

A note on protocol, because the details change how you read the table. OpenAI is requested at 384 through its native dimensions parameter. The Latimal row is truncated to 384 the same way as the open-weight rows; its public API also serves 384 directly through a dimension parameter, which is what the reproduction script uses. Voyage is requested at 512, the narrowest width its API serves above 384, then truncated to 384 and re-normalized; every other dense model, Cohere included, is truncated to its first 384 dimensions and re-normalized. Truncation can understate a model's native-dimension quality, so read each row as that model's d=384 operating point, a width production deployments commonly pay for.

Cohere and Voyage are called with their retrieval input types (queries as search queries, everything else as documents); Cohere's classification and clustering input types are not used, so every text passes through each API's retrieval treatment rather than a per-task one. Open-weight models are encoded as raw text with no instruction prefixes throughout, including models whose cards recommend or require them (multilingual-e5-large, Nomic embed-text v1.5); both cards prescribe prefixes for classification and clustering too, so those rows can be understated beyond retrieval. Prefixed runs are welcome as separate submissions (the package ships a `PromptedAdapter` wrapper for exactly this).

Latimal food-embed-v1 is our model. We publish the benchmark and hold a row on it. The row is measured locally with the same harness, encoded as raw text with no instruction prefixes, exactly like the open-weight rows; the public API is the reproduction route. The sub-tables below show where it loses: Voyage, OpenAI, and Cohere embed-v4 all outscore it on Search; OpenAI and Voyage outscore it on Classification. The no-training rule binds our row too. We screen our training corpora against the benchmark data for overlap, and the removal policy in [CONTRIBUTING](https://github.com/latimal/foodeval/blob/main/CONTRIBUTING.md) applies to our row as it does to any submission. Reproduce the row with [`scripts/run_latimal.py`](https://github.com/latimal/foodeval/blob/main/scripts/run_latimal.py) and an API key from [latimal.com](https://latimal.com) (self-serve; the free credit grant covers the full suite about ten times over). That script is the custom-adapter route for any model without a built-in adapter; its one caveat is in the reproducer notes below.

### Overall

| Rank | Model | Dim | Search | Matching | Classification | **FoodEval Score** |
|------|-------|----:|-------:|---------:|---------------:|-------------------:|
| 1 | Latimal food-embed-v1 | 384 | 0.4783 | 0.8513 | 0.7382 | **0.7176** |
| 2 | OpenAI text-embedding-3-large | 384 | 0.5535 | 0.7580 | 0.8329 | **0.6961** |
| 3 | Voyage 4 Large | 384 | 0.5577 | 0.7409 | 0.7895 | **0.6838** |
| 4 | Cohere embed-v4 | 384 | 0.5173 | 0.7407 | 0.7369 | **0.6659** |
| 5 | Nomic embed-text v1.5 | 384 | 0.4362 | 0.7390 | 0.7103 | **0.6357** |
| 6 | Alibaba GTE-large-en v1.5 | 384 | 0.4741 | 0.6993 | 0.7156 | **0.6256** |
| 7 | BAAI/bge-m3 | 384 | 0.4156 | 0.7182 | 0.7009 | **0.6159** |
| 8 | Lexical (TF) | 4096 | 0.2852 | 0.7278 | 0.6885 | **0.5770** |
| 9 | multilingual-e5-large | 384 | 0.3936 | 0.7038 | 0.3986 | **0.5750** |
| 10 | Cohere multilingual-v3 | 384 | 0.3897 | 0.6823 | 0.5063 | **0.5701** |

_multilingual-e5-large and Nomic embed-text v1.5 run without the instruction prefixes their model cards prescribe; see the protocol note above._

### Search
_NDCG@10, 4 tasks._

| Rank | Model | Food | Concept | Diet | Noisy | **Avg** |
|------|-------|------:|------:|------:|------:|--------:|
| 1 | Voyage 4 Large | 0.6779 | 0.5615 | 0.2378 | 0.7535 | **0.5577** |
| 2 | OpenAI text-embedding-3-large | 0.6905 | 0.5503 | 0.2163 | 0.7569 | **0.5535** |
| 3 | Cohere embed-v4 | 0.6439 | 0.4941 | 0.1829 | 0.7482 | **0.5173** |
| 4 | Latimal food-embed-v1 | 0.6133 | 0.4348 | 0.2006 | 0.6646 | **0.4783** |
| 5 | Alibaba GTE-large-en v1.5 | 0.6020 | 0.4686 | 0.2273 | 0.5986 | **0.4741** |
| 6 | Nomic embed-text v1.5 | 0.5750 | 0.3661 | 0.1577 | 0.6461 | **0.4362** |
| 7 | BAAI/bge-m3 | 0.5529 | 0.3364 | 0.1483 | 0.6247 | **0.4156** |
| 8 | multilingual-e5-large | 0.5364 | 0.3164 | 0.1385 | 0.5831 | **0.3936** |
| 9 | Cohere multilingual-v3 | 0.5134 | 0.3483 | 0.1371 | 0.5602 | **0.3897** |
| 10 | Lexical (TF) | 0.5353 | 0.2011 | 0.0887 | 0.3157 | **0.2852** |

### Matching
_Best F1, 7 tasks._

| Rank | Model | Indian | Global | Bev | Bakery | Portion | Noisy Menu | X-Lingual | **Avg** |
|------|-------|------:|------:|------:|------:|------:|------:|------:|--------:|
| 1 | Latimal food-embed-v1 | 0.8165 | 0.8673 | 0.7461 | 0.7545 | 0.9722 | 0.9164 | 0.8862 | **0.8513** |
| 2 | OpenAI text-embedding-3-large | 0.7454 | 0.8284 | 0.7149 | 0.7349 | 0.8493 | 0.6850 | 0.7484 | **0.7580** |
| 3 | Voyage 4 Large | 0.7178 | 0.7831 | 0.7192 | 0.7154 | 0.7909 | 0.6400 | 0.8196 | **0.7409** |
| 4 | Cohere embed-v4 | 0.7316 | 0.8288 | 0.7095 | 0.6914 | 0.8354 | 0.6667 | 0.7214 | **0.7407** |
| 5 | Nomic embed-text v1.5 | 0.7305 | 0.7320 | 0.7153 | 0.6838 | 0.8548 | 0.7500 | 0.7069 | **0.7390** |
| 6 | Lexical (TF) | 0.6867 | 0.6868 | 0.7061 | 0.6824 | 0.8044 | 0.8215 | 0.7069 | **0.7278** |
| 7 | BAAI/bge-m3 | 0.7107 | 0.7160 | 0.7061 | 0.6837 | 0.8207 | 0.6736 | 0.7169 | **0.7182** |
| 8 | multilingual-e5-large | 0.6805 | 0.7157 | 0.7061 | 0.6879 | 0.7572 | 0.6482 | 0.7310 | **0.7038** |
| 9 | Alibaba GTE-large-en v1.5 | 0.7048 | 0.6950 | 0.7098 | 0.6824 | 0.7246 | 0.6715 | 0.7069 | **0.6993** |
| 10 | Cohere multilingual-v3 | 0.6941 | 0.6711 | 0.7061 | 0.6824 | 0.6687 | 0.6400 | 0.7136 | **0.6823** |

### Classification
_Macro F1, 1 task._

| Rank | Model | Cuisine | **Avg** |
|------|-------|------:|--------:|
| 1 | OpenAI text-embedding-3-large | 0.8329 | **0.8329** |
| 2 | Voyage 4 Large | 0.7895 | **0.7895** |
| 3 | Latimal food-embed-v1 | 0.7382 | **0.7382** |
| 4 | Cohere embed-v4 | 0.7369 | **0.7369** |
| 5 | Alibaba GTE-large-en v1.5 | 0.7156 | **0.7156** |
| 6 | Nomic embed-text v1.5 | 0.7103 | **0.7103** |
| 7 | BAAI/bge-m3 | 0.7009 | **0.7009** |
| 8 | Lexical (TF) | 0.6885 | **0.6885** |
| 9 | Cohere multilingual-v3 | 0.5063 | **0.5063** |
| 10 | multilingual-e5-large | 0.3986 | **0.3986** |

### Reproduce Any Row

The exact command for every row. Run these from a checkout of the GitHub repo after `pip install -e ".[all]"`; the notes below cover where a rerun can differ from the published numbers. Export `OPENAI_API_KEY`, `COHERE_API_KEY`, and `VOYAGE_API_KEY` first for the API rows (see Running the Benchmark). Local models are deterministic on a given hardware and library stack. API rows are point-in-time: each result file carries its evaluation timestamp, and reproduction depends on the provider serving the same model it served then.

```bash
foodeval run --model lexical-tf --output results/bm25.json  # legacy filename, kept to match the shipped file
foodeval run --model BAAI/bge-m3 --dim 384 --output results/bge_m3_384.json
foodeval run --model Alibaba-NLP/gte-large-en-v1.5 --dim 384 --output results/gte_large_v15_384.json
foodeval run --model intfloat/multilingual-e5-large --dim 384 --output results/e5_large_384.json
foodeval run --model nomic-ai/nomic-embed-text-v1.5 --dim 384 --output results/nomic_embed_v15_384.json
foodeval run --model openai:text-embedding-3-large --dim 384 --output results/openai_te3_large_384.json
foodeval run --model cohere:embed-v4.0 --dim 384 --output results/cohere_embed_v4_384.json
foodeval run --model cohere:embed-multilingual-v3.0 --dim 384 --output results/cohere_multilingual_v3_384.json
foodeval run --model voyage:voyage-4-large --dim 384 --output results/voyage_4_large_384.json
LATIMAL_API_KEY=... python3 scripts/run_latimal.py --dim 384 --output results/latimal_food_embed_v1_384.json
```

Three notes for reproducers. The published local rows were measured with sentence-transformers 5.3.0, torch 2.10.0, and scikit-learn 1.8.0 on Apple Silicon (MPS); if your numbers differ in the last decimals, check those versions first. Shipped result files carry curated display names, and their timings reflect cached re-aggregation rather than wall-clock runs; a rerun's `model_name` will be the adapter's raw name. Only the scores need to match, with one exception. For the Latimal row, the production /embed endpoint applies standard input normalization, so an API rerun scores above the published row wherever inputs carry markup noise. In our verification run the aggregate came back +0.012 (task deltas -0.004 to +0.037, the largest on Bakery Match). The published row was measured on raw text, like every other row.

To submit your model's results, see [Contributing Results](https://github.com/latimal/foodeval#contributing-results).

## Task Descriptions

### Food Search

178 queries across three menu domains (Indian, global, beverage) matched against a shared corpus of 974 items. Each query has graded relevance judgments (0-3), assigned by the Latimal team and audited in multiple passes (see Data Provenance). This is the core search quality task. Category queries like "appetizers and starters" require knowing what counts as one: "arancini," "bruschetta," and "edamame" are all relevant, and general web-text models cluster them only loosely. Even the strongest model on this task leaves headroom (top score 0.69 NDCG@10).

### Concept Search

44 abstract concept queries ("barbecue dish," "biryani and rice dish," "warm comfort food," "crispy appetizer") matched against 301 corpus items. Concept search tests whether the model understands food categories or just matches on lexical overlap. A model that pairs "biryani" with "chicken biryani" on substring overlap alone will fail when "warm comfort food" has to map to "mac and cheese" or "rajma."

### Diet/Allergen Search

35 queries centered on diet and allergen constraints ("celiac friendly," "shellfish allergy safe," "halal food"), plus a few nutritional-property and meal-type queries ("high protein," "iron rich dishes," "breakfast options"), matched against 245 items. Requires the model to understand dietary properties that are rarely stated explicitly in menu text. "Grilled salmon" is, as typically prepared, gluten-free; the model has to know that on its own, since nothing in the item name says so. Relevance grades reflect typical preparations: the task measures retrieval under a dietary constraint. It does not certify allergen safety.

### Noisy Search

46 queries with realistic misspellings and abbreviations ("bbq burgeer," "bibimbab," "paner tikka") matched against 304 corpus items. Real users do not type menu item names correctly. This task measures robustness to the kind of input a production search bar actually receives. The Lexical (TF) baseline manages only 0.316 NDCG@10 here, less than half what the leading dense models score on the same queries; subword-tokenized models handle noise much better.

### Indian Match

498 menu-item pairs drawn from Indian-anchored production menus, which also carry the pizza, fast-food, and cafe items real menus do. Tests same-dish detection for pairs like "Chef's Special Biryani" vs. "Biryani" (same dish) and "Chicken Tikka" vs. "Paneer Tikka" (different dishes, confusable names).

### Global Match

318 pairs of non-Indian cuisine menu items spanning global cuisines. Tests same-dish detection for pairs like "Pad Thai Chicken" vs. "Chicken Pad Thai" (same dish) and "Gyoza" vs. "Shumai" (different dumplings, same category). Global food items have fewer transliteration variants but more cross-cuisine confusables.

### Beverage Match

295 pairs of beverage items including teas, coffees, juices, smoothies, and cocktails. Beverages have their own deduplication challenges: "Aam Lassi" vs. "Mango Lassi" (same drink, bilingual naming), "Acai Smoothie" vs. "Acai Bowl" (same base, different product). Modifier sensitivity matters more for beverages than for food.

### Bakery Match

336 pairs of bakery and dessert items. Tests same-item detection for pairs like "Almond Croissant" vs. "Croissant Almond" (same item, reordered) and "Apple Crumble" vs. "Apple Cider" (different items, shared prefix). Bakery items have high within-category similarity that produces false merges.

### Portion Size

228 pairs testing whether the model can distinguish portion or size variants from genuinely different items. Examples include "Regular Pepsi" vs. "Large Pepsi" (same item, different size) and "Large Coffee" vs. "Large Smoothie" (same size word, different items). Models must learn that size modifiers do not change dish identity, while ingredient and preparation modifiers do.

### Noisy Menu Match

323 pairs with markup noise, casing artifacts, and symbol clutter. Tests robustness to real-world menu formatting like `***HOT*** Spicy Ramen (Large) @¥980 🌶` vs. `Spicy Ramen` and `#7 BUTTER CHICKEN 🔥 $14.99 [BESTSELLER]` vs. `Butter Chicken` (same dish behind the decoration). Production menus are messy; models have to see through the noise.

### Cross-Lingual Match

514 pairs across three categories: romanized (186 pairs), bilingual (173), and cross-script/CJK (155). Tests whether the model can match dishes across writing systems and transliterations. Examples include `Beef Pho` vs. `Pho Bo` (romanized), `Aloo Gobi आलू गोभी` vs. `आलू गोभी` (bilingual), and cross-script pairs mixing Latin, Devanagari, CJK, and other scripts. General multilingual models often key on "both are cross-lingual food text" rather than actual dish identity, producing false merges.

### Cuisine Classification

3,053 items across 26 menu-taxonomy classes: mostly cuisines (North Indian, Italian, Mexican, Japanese), plus format categories like QSR and Street Food that real menu systems must handle. The evaluation trains a LogisticRegression probe on frozen embeddings with an 80/20 stratified split, repeated across 10 random seeds. Macro F1 is reported. This measures how well the embedding space separates cuisines. Low-frequency classes like SE Asian, Goan, and Ethiopian are deliberately included with only 30 to 35 examples each, because real menus contain these cuisines and a production system has to cope with them.

## Data Format

FoodEval stores all data as JSON files in `foodeval/data/`. Three schemas correspond to the three task types.

### Retrieval Tasks (food_search, concept_search, diet_search, noisy_search)

```json
{
  "task": "food_search",
  "version": "0.1.0",
  "description": "...",
  "metric": "ndcg@10",
  "corpus": ["butter chicken", "paneer tikka", "..."],
  "queries": [
    {
      "id": "q001",
      "query": "american burger and fries",
      "domain": "global",
      "relevance": {
        "cheeseburger": 3,
        "hot dog": 2,
        "bbq ribs": 1
      }
    }
  ],
  "metadata": {
    "n_queries": 178,
    "n_corpus": 974,
    "domains": ["beverage", "global", "indian"]
  }
}
```

Relevance grades: 3 = highly relevant, 2 = relevant, 1 = marginally relevant. Grade-0 items are omitted from the relevance map; anything absent from a query's map scores 0, the standard convention for incompletely judged collections. Suspected missing judgments are handled as versioned data fixes (see [Report a data issue](https://github.com/latimal/foodeval/blob/main/CONTRIBUTING.md#report-a-data-issue) in CONTRIBUTING).

### Pair Classification Tasks (indian_match, global_match, beverage_match, bakery_match, portion_size, noisy_menu_match, cross_lingual_match)

```json
{
  "task": "indian_match",
  "version": "0.1.0",
  "description": "...",
  "metric": "best_f1",
  "pairs": [
    {
      "id": "p001",
      "text_a": "**BESTSELLER** Butter Chicken",
      "text_b": "Butter Chicken",
      "label": 1
    }
  ],
  "metadata": {
    "n_pairs": 498,
    "n_positive": 234,
    "n_negative": 264
  }
}
```

Labels: 1 = same dish, 0 = different dish. `cross_lingual_match` pairs additionally carry a `domain` field (`romanized`, `bilingual`, or `cross_script`), echoed in its metadata as `domains`.

### Classification Task (cuisine_classify)

```json
{
  "task": "cuisine_classify",
  "version": "0.1.0",
  "description": "...",
  "metric": "macro_f1",
  "items": [
    {
      "id": "i001",
      "text": "aaloo gobi",
      "label": "North Indian",
      "source": "indian"
    }
  ],
  "label_names": ["American", "Bengali", "Biryani", "..."],
  "metadata": {
    "n_items": 3053,
    "n_classes": 26
  }
}
```

## Running the Benchmark

### Local Models (sentence-transformers)

```bash
pip install "foodeval[local]"

# Run all tasks
foodeval run --model BAAI/bge-m3 --dim 384

# Run specific tasks
foodeval run --model BAAI/bge-m3 --dim 384 --tasks food_search,indian_match

# Save results to JSON
foodeval run --model BAAI/bge-m3 --dim 384 --output results/bge_m3_384.json

# Local model directory
foodeval run --model ./my-fine-tuned-model --dim 384
```

Note: the sentence-transformers adapter loads models with `trust_remote_code=True` (some architectures require it, e.g. GTE v1.5). Point it only at checkpoints you trust.

### OpenAI Models

```bash
pip install "foodeval[api]"
export OPENAI_API_KEY=sk-...

foodeval run --model openai:text-embedding-3-large --dim 384
```

### AWS Bedrock Models

```bash
pip install "foodeval[api]"

foodeval run --model bedrock:cohere.embed-multilingual-v3 --dim 384
foodeval run --model bedrock:amazon.titan-embed-text-v2:0 --dim 384
```

Uses your standard AWS credential chain (env vars, `~/.aws/credentials`, or an instance role). Default region is `us-east-1`; override with `AWS_BEDROCK_REGION`.

### Cohere and Voyage

Both adapters use the providers' REST APIs directly, so the base install is enough.

```bash
export COHERE_API_KEY=...
foodeval run --model cohere:embed-v4.0 --dim 384

export VOYAGE_API_KEY=...
foodeval run --model voyage:voyage-4-large --dim 384
```

The Voyage adapter paces requests for the free tier (3 requests per minute, about 21 seconds between calls; a full run takes roughly 30 minutes). Set `VOYAGE_MIN_INTERVAL=0` for paid keys. The default endpoint is the MongoDB-hosted Voyage service; override with `VOYAGE_BASE_URL`.

### Gemini and Vertex AI

Gemini and Vertex AI also run on the base install, calling Google's REST endpoints directly. Vertex authenticates with Application Default Credentials (`gcloud auth application-default login`).

```bash
export GEMINI_API_KEY=...
foodeval run --model gemini:gemini-embedding-2 --dim 384

export GOOGLE_CLOUD_PROJECT=...
foodeval run --model vertex:gemini-embedding-001 --dim 384
```

### Lexical Baseline (TF)

```bash
pip install foodeval
foodeval run --model lexical-tf
```

The CLI key is `lexical-tf` (`bm25` is accepted as a legacy alias); the implementation is hashed term-frequency vectors with cosine similarity, deliberately minimal (no IDF, no BM25-style length weighting). It answers one question: does plain term overlap solve the task?

### Other Commands

```bash
# List all available tasks
foodeval list

# Inspect a task
foodeval info food_search

# Generate leaderboard from saved result files
foodeval leaderboard results/

# Check a training corpus for exact overlap with benchmark data
foodeval preflight --compare path/to/training_data/

# Plan or execute the baseline run matrix
foodeval matrix --list
```

### Custom Adapters

Implement the `EmbeddingAdapter` protocol to evaluate any model:

```python
import numpy as np
from foodeval.evaluate import run_benchmark

class MyAdapter:
    @property
    def name(self) -> str:
        return "my-model-384d"

    @property
    def dimension(self) -> int:
        return 384

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        # Your encoding logic here
        # Return float32 array of shape (len(texts), self.dimension)
        ...

result = run_benchmark(MyAdapter())
result.to_json("results/my_model.json")
```

## Evaluation Methodology

### Metrics

- **NDCG@10** for retrieval tasks. Normalized Discounted Cumulative Gain at rank 10, computed from graded relevance judgments with exponential gains (2^rel - 1) and log2 position discounts; the normalizer is the ideal DCG at the same rank-10 cut, taken over the query's complete relevance set (the standard definition), so a perfect ranking scores 1.0 on every query. All corpus items are ranked by cosine similarity against the query embedding. Mean NDCG@10 across all queries is the primary score.
- **Best F1** for pair classification tasks. Cosine similarity is computed for each pair. A threshold sweep finds the operating point that maximizes F1 score. The sweep uses the actual observed similarity values as threshold candidates, guaranteeing the true optimal threshold is found. Two properties to keep in mind when reading Best F1. First, because the threshold is chosen on the evaluation pairs themselves, Best F1 is an oracle operating point and an upper bound on deployed F1; choosing the threshold this way is standard practice for pair classification. Second, the metric has a floor: predicting every pair positive yields 2p/(1+p), where p is the positive rate; on these tasks that floor ranges from 0.639 (Indian Match) to 0.707 (Cross-Lingual Match). A score at the floor (several appear in the leaderboard, e.g. 0.7061 on Beverage Match) means no similarity threshold separates same-dish from different-dish pairs better than calling every pair a match.
- **Macro F1** for classification. A LogisticRegression probe is trained on frozen embeddings with an 80/20 stratified split. The experiment is repeated across 10 random seeds (42-51). Mean macro F1 across seeds is reported along with standard deviation.

### Aggregate Score

FoodEval Score is the unweighted mean of all 12 task main scores (1/12 each), stored as `aggregate_score` in result JSONs. That makes Matching, with 7 of the 12 tasks, the largest contributor to the composite. This is the flat-mean convention the original MTEB leaderboard popularized. It keeps the composite trivially recomputable: take the mean of the 12 `main_score` values. Hand-averaging the printed category cells can disagree in the last digit due to rounding. Category means for Search (mean NDCG@10 across 4 tasks), Matching (mean Best F1 across 7 tasks), and Classification (Macro F1) are reported alongside for capability-level comparison. Tasks have different scales (Best F1 has an all-positive floor of 0.639 to 0.707 on these tasks, while NDCG@10 on diet and allergen search tops out below 0.24), so the aggregate is best used for relative ranking rather than absolute interpretation.

### Confidence Intervals

For retrieval tasks, 95% confidence intervals come from bootstrap resampling (1,000 iterations) of per-query NDCG@10. For matching tasks, pairs are resampled and F1 is recomputed at the chosen threshold. Classification reports mean and standard deviation across the 10 seeds instead. These appear in the detailed JSON output but are omitted from the leaderboard tables for readability.

### Data Provenance

All items are individual menu-item names drawn from production menu data, then curated, relabeled, and deduplicated by the Latimal team. Relevance grades, pair labels, and cuisine labels were produced in-house and audited in multiple passes: label fixes, relevance-grade recalibration, and duplicate removal. The canonical copy of the dataset lives on the Hub at [huggingface.co/datasets/latimal/foodeval](https://huggingface.co/datasets/latimal/foodeval) (gated, agreement auto-approved). Labeling disputes surfaced by the community are handled through versioned data fixes: each task file carries a `version` field, and any fix large enough to move scores is flagged in the changelog (see [Report a data issue](https://github.com/latimal/foodeval/blob/main/CONTRIBUTING.md#report-a-data-issue)).

### Reproducibility

All evaluation data is fixed and ships with the package. Every stochastic component is seeded: the classification probe (10 fixed seeds, 42-51) and the bootstrap confidence intervals (fixed resampling seed), so repeat runs are stable given the same model weights and adapter configuration.

## FAQ

### Why food embeddings?

Food delivery platforms process millions of menu items across dozens of languages and writing systems. The core operations (search, deduplication, classification, recommendation) all depend on text embeddings. General-purpose models trained on web text miss food-specific semantics: that "Dal Makhani" and "Black Lentil Curry" are the same dish, that "warm comfort food" should return "mac and cheese" and "tomato soup," or that a menu item with `***HOT***` prefixed is the same dish as the one without it. These gaps cause real production failures.

### Why not just use MTEB?

MTEB is the standard for general embedding evaluation. It contains no menu- or dish-level tasks (the closest, NFCorpus, retrieves medical literature for nutrition-seeded queries). Every model on our leaderboard, our own and the strongest general-purpose API models included, scores below 0.24 NDCG@10 on diet and allergen search, and two open-weight models sit at the all-positive floor on cross-lingual matching. FoodEval is complementary to MTEB: it measures domain-specific capabilities that general benchmarks do not surface.

### Can I submit results?

Yes. Run the full benchmark at d=384, save the output JSON, and open a pull request with your result file; the steps are in [Contributing Results](https://github.com/latimal/foodeval#contributing-results).

### What dimension should I use?

The leaderboard standardizes on d=384 so every model is compared at the same width. This is a practical operating point for production deployment (low memory, fast cosine search). On the board, every dense model runs at 384 regardless of whether it was trained for truncation (the protocol note above the leaderboard spells out how). Outside the leaderboard you can evaluate at any dimension by passing `--dim` to the CLI, or omit it for the model's native width.

### Can I use this data for training?

No. FoodEval is licensed under CC-BY-NC-4.0 with an evaluation-only addendum. Using this data as training data for machine learning models is expressly prohibited. Without that restriction, the scores stop meaning anything.

## Contributing Results

1. Run the full benchmark (all 12 tasks) at d=384 for leaderboard inclusion.
2. Save results: `foodeval run --model your-model --dim 384 --output results/your_model_384.json`.
3. Open a PR to [github.com/latimal/foodeval](https://github.com/latimal/foodeval) with the result JSON.
4. Include the model name, source (Hugging Face ID, API, etc.), and any notable configuration.
5. Confirm in the PR that the model was not trained, fine-tuned, or distilled on FoodEval data. Full requirements: [CONTRIBUTING.md](https://github.com/latimal/foodeval/blob/main/CONTRIBUTING.md).

## Citation

```bibtex
@misc{foodeval2026,
  title   = {FoodEval: A Benchmark for Food-Domain Text Embeddings},
  author  = {Patni, Aditya},
  year    = {2026},
  url     = {https://github.com/latimal/foodeval},
  version = {0.1.0},
  license = {CC-BY-NC-4.0 with evaluation-only addendum}
}
```

## License

CC-BY-NC-4.0 with an evaluation-only addendum. See [LICENSE](https://github.com/latimal/foodeval/blob/main/LICENSE) for the license summary and addendum text, and [creativecommons.org/licenses/by-nc/4.0/legalcode](https://creativecommons.org/licenses/by-nc/4.0/legalcode) for the full legal code.

As licensor, Latimal grants two permissions on top of the license, recorded in [LICENSE](https://github.com/latimal/foodeval/blob/main/LICENSE): running FoodEval to evaluate and compare models is permitted in any setting, including inside a company; and the evaluation code and tooling may be used commercially. What is never permitted, in any setting: using the data to train, fine-tune, or distill a model, or selling the data. The data may be redistributed only under these same terms, unmodified; the code carries the commercial-use permission above.
