# Changelog

## 0.1.0 (2026-06-11)

Initial public release.

- 12 tasks: 4 search (NDCG@10), 7 matching (Best F1), and 26-class cuisine classification (Macro F1); 5,868 evaluation examples built from production menu data across multiple writing systems
- 10-model leaderboard (nine dense models at d=384 plus a lexical baseline); FoodEval Score is the unweighted mean across all 12 tasks
- Adapters for sentence-transformers (local and HF Hub), OpenAI, Cohere, Voyage, Gemini, Vertex AI, AWS Bedrock, plus a lexical term-frequency baseline
- CLI: `foodeval run / list / info / leaderboard / preflight / matrix` (preflight checks training-data overlap; matrix plans or executes baseline runs)
- Licensed CC-BY-NC-4.0 with an evaluation-only addendum (no training on the data)
