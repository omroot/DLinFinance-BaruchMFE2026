# Deep Learning in Finance (Baruch MFE)

This repository hosts the lecture material, notebooks, homework, and final project for the Baruch College MFE elective **Deep Learning in Finance (Fall 2026)**.

**Material is published week by week.** Weeks 1-2 and the final project are available now; later weeks appear here as we reach them. See the [course roadmap](#course-roadmap) for what is coming.

## Course Syllabus
- Download the full syllabus from [`Syllabus/DeepLearningInFinance_Syllabus_BaruchMFE_Fall2026.pdf`](Syllabus/DeepLearningInFinance_Syllabus_BaruchMFE_Fall2026.pdf).
- **Grading.** Research paper presentation 40%, final project 60%. Weekly homeworks are optional and ungraded — they are provided as practice and as preparation for the final project.
- **Paper presentation.** The Assignment 1 handout is at [`Assignments/DeepLearningInFinance_Assignment1_PaperPresentation_BaruchMFE_Fall2026.pdf`](Assignments/DeepLearningInFinance_Assignment1_PaperPresentation_BaruchMFE_Fall2026.pdf).
- **Changes for Fall 2026.** The asset pricing papers that previously filled Week 7 are now distributed to the week whose architecture they use: the virtue of complexity work moves to Week 1, deep learning statistical arbitrage to Week 2, and deep learning in asset pricing to Week 4. Week 3 gains neural Hawkes processes, and Week 7 is now Bayesian deep networks for factor modeling.

## Week 1 — Theoretical Foundations and the Virtue of Complexity

Approximation power of deep versus shallow networks, the Universal Approximation Theorem, double descent and overparameterization, scaling laws, and the spectral view of trained weights.

- **Slides.** [`lecture_slides/week1/TheoreticalFoundation.pdf`](lecture_slides/week1/TheoreticalFoundation.pdf) — Theoretical Foundations of Deep Learning, Parts I–III (101 slides).
- **Theory notebooks** in [`src/notebooks/theory/`](src/notebooks/theory/):
  - [approximation.ipynb](src/notebooks/theory/approximation.ipynb) — compositional functions and the depth advantage
  - [double_descent.ipynb](src/notebooks/theory/double_descent.ipynb) — the double descent risk curve
  - [scaling_laws.ipynb](src/notebooks/theory/scaling_laws.ipynb) — how error falls with model and data size
  - [spectral_analysis.ipynb](src/notebooks/theory/spectral_analysis.ipynb) — weight spectra of a fine-tuned language model
- **Virtue of complexity.** The double descent and overparameterization material closes with Kelly et al., *The Virtue of Complexity in Return Prediction*. Replication code: [TheVirtueOfComplexity_PaperReplication](https://github.com/omroot/TheVirtueOfComplexity_PaperReplication).
- **Homework (optional).** Start with [`homeworks/week1/hw01.md`](homeworks/week1/hw01.md); the three notebooks sit alongside it.

The theory notebooks import their helper functions from `src/dlf/theory/`, so run them from inside `src/notebooks/theory/` — each one adds the repository root to `sys.path` on its first cell. The homework notebooks are self-contained and need no course code.

## Week 2 — Transformers

Attention and the Transformer architecture, applied to financial time series, to tabular data, and to statistical arbitrage.

- **Slides** in [`lecture_slides/week2/`](lecture_slides/week2/):
  - [Part I — Attention & Transformers](lecture_slides/week2/Transformers_AttentionFoundations_Part1.pdf) (41 slides)
  - [Part II — Transformers for Financial Time Series](lecture_slides/week2/Transformers_FinTSForecasting_Part2.pdf) (26 slides)
  - [Part III — Transformers for Tabular Data](lecture_slides/week2/Transformers_StructuredData_Part3.pdf) (42 slides)
  - [Part IV — Deep Learning Statistical Arbitrage](lecture_slides/week2/Transformers_StatArb_Part4.pdf) (45 slides)
- **Statistical arbitrage.** Part IV follows Guijarro-Ordonez, Pelger & Zanotti, *Deep Learning Statistical Arbitrage*, as a sequence-model application. Replication code: [DeepLearningStatisticalArbitrage_PaperReplication](https://github.com/omroot/DeepLearningStatisticalArbitrage_PaperReplication).

The transformer illustration notebooks and the optional homework for this week will be published shortly.

## Course Roadmap

Remaining weeks, published as we reach them:

| Week | Topic |
|------|-------|
| 3 | Neural ODEs and neural point processes (neural Hawkes, jump SDEs) |
| 4 | Generative Adversarial Networks, and deep learning in asset pricing |
| 5 | Diffusion models and score-based generative models |
| 6 | Graph Neural Networks |
| 7 | Bayesian deep networks for factor modeling |

Week 7 follows Matthew Dixon and Nicholas Polson, *Bayesian Deep Fundamental Factor Models*, Chapter 32 in Capponi & Lehalle (eds.), [*Machine Learning and Data Sciences for Financial Markets*](https://www.cambridge.org/core/books/abs/machine-learning-and-data-sciences-for-financial-markets/bayesian-deep-fundamental-factor-models/DEE56793B55116E1D68BB72848C7D7B4) (Cambridge University Press, 2023), together with their earlier [*Deep Fundamental Factor Models*](https://arxiv.org/abs/1903.07677).

## Final Project — Deep Learning Enhanced Systematic Equity Strategy

The capstone asks you to take a complete, working systematic equity strategy and improve it with deep learning, then analyze honestly whether it helped. It is worth 60% of the grade, so start reading early even though the deep learning weeks are still ahead.

- **Assignment:** [`final_project/final_project_assignment.md`](final_project/final_project_assignment.md) — requirements, deliverables, and the grading rubric.
- **Baseline pipeline:** [`final_project/systematic_equity_strategy.ipynb`](final_project/systematic_equity_strategy.ipynb) — Fama-French 5-factor alpha estimation, 21 engineered technical features, ML alpha models (ElasticNet / XGBoost / RandomForest), portfolio optimization with PyPortfolioOpt, and walk-forward backtesting with transaction costs.
- **Data:** S&P 500 daily adjusted closes with sector metadata ([`final_project/sp500_prices_with_metadata.parquet.zip`](final_project/sp500_prices_with_metadata.parquet.zip) — unzip before use), GICS sectors ([`final_project/sp500_metadata.csv`](final_project/sp500_metadata.csv)), and [`final_project/FamaFrench_5_Factors_daily_returns.csv`](final_project/FamaFrench_5_Factors_daily_returns.csv). Restrict your analysis to the **top 100 names** by market cap or average volume to keep run times manageable.
- **Where deep learning comes in:** a deep alpha model (GNN, cross-sectional or temporal Transformer, LSTM/GRU/TCN, Neural ODE) and/or a deep risk model (diffusion- or autoencoder-based covariance denoising). You must use at least one.

**Beating the baseline is not required for a strong grade.** Methodological rigor — no look-ahead, honest validation, and a clear account of why something did or did not work — carries more weight than the Sharpe ratio you land on.

Note: the large raw vendor extracts, the derived price matrix, and the reference performance CSVs are not tracked in git. Ask if you want the reference numbers for comparison.

## Repository Layout
- `homeworks/`: optional weekly practice notebooks and starter code. Solutions are distributed separately.
- `final_project/`: capstone assignment, baseline strategy notebook, and project datasets.
- `src/`: Python modules and interactive notebooks (see below).
- `Syllabus/`, `Assignments/`: syllabus and the paper presentation handout.

### `src/` structure
- `src/dlf/`: reusable course library code.
  - `theory/`: helper functions behind the Week 1 theory notebooks.
  - `models/`: model implementations used in later weeks.
  - `preprocess/`: data preparation utilities.
- `src/notebooks/`: illustrative notebooks used to demystify concepts discussed in class, organized by topic. `theory/` is published; the remaining topics appear with their week.

## Getting Started
1. Clone the repository: `git clone https://github.com/omroot/DLinFinance-BaruchMFE2026.git`
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
3. Install the shared dependencies: `pip install -r requirements.txt`
4. A few notebooks need extras beyond the shared file — install them as you reach the relevant week:
   - `weightwatcher` (Week 1, spectral analysis and homework part 3)
   - `torchdiffeq` (Week 3, Neural ODEs)
   - `torch-geometric` (Week 6, GNNs)
   - `PyPortfolioOpt`, `xgboost`, `pyarrow` (final project)

## Use of AI in Course Material Preparation

AI tools were used to assist in the preparation of this course material, primarily for code refactoring and polishing the presentation of homework and illustration notebooks. The instructor wrote the original code and concepts, and AI was employed to improve code quality, consistency, and presentation.

**The following remain the sole responsibility of the instructor:**
- Selection and sequencing of topics
- Conceptual content and pedagogical approach
- Lecture structure and ordering
- Preparation of all lecture slides
- Academic integrity and accuracy of course material

Lecturer: **Oualid Missaoui** — reach out at `oualid.missaoui@baruch.cuny.edu` or during office hours.
